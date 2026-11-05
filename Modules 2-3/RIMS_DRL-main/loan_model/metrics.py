"""
Resilience measure for the stage-model simulator (Phase 4).

Q(t) = share of open cases that are on track: a case is on track while its age is at
most the deadline D_k of the stage it is in (D_k = p90 of case age at stage exit in
the log, floored at `deadline_floor_s`). Q(t) is piecewise constant: it changes when a
case arrives, changes stage, ends, or its age crosses its stage deadline.

Loss = integral of (1 - Q(t)) dt: every moment a case is off track counts, with no
reference level or threshold (an earlier version only counted Q below Q_ref, the 5th
percentile of normal days, which made about half of all deadline misses free). It is
integrated exactly: deadline crossings are kept in a heap and processed lazily in time
order whenever the tracker is advanced, so no simulation events are needed for them.
Normal operation (Q ~ 0.90) also accrues Loss; what a disruption adds is measured
against its undisrupted twin (calibrate_loan_resilience.py --scenarios).

Recovery (recovered(), reported only, not part of the Loss or the reward): the
disruption is over, every step it blocked has been carried out (the disruption layer
calls clear_hit(); leaving the stage or ending also clears a hit), and the rolling mean
of Q over the last 24 h is back at `recover_q` (the mean Q of undisrupted runs) or above.

Per-case shares (case_loss(), for the per-case reward of loan_env): at every instant the
Loss rate 1 - Q = off/open is split equally among the cases that are off track, so each
off-track case accrues dt/open and the shares of all cases add up to the Loss exactly.
A running integral G(t) of the rate per off-track case is kept; a case accrues dG while
it is off track. A case that expires is charged dG from its expiry on (as if it stayed
off track): otherwise its leaving the open set would make an expiry look free.

Deadline mode 'age' (default, the agreed definition) measures case age against D_k;
'sojourn' (time in the current stage against a stage-duration deadline) was compared
in Phase 4 and had a lower signal-to-noise ratio, so it is kept only as an option.

The tracker is an observer of loan_model.simulator.Simulator (on_arrival, on_stage,
on_end, on_expiry); call advance(t) before reading Q, the Loss or recovery.
"""
import heapq
from collections import deque


class QTracker:
    def __init__(self, deadlines, end_stages, recover_q=None, deadline_floor_s=86400.0, mode='age'):
        self.deadline = {s: max(float(d), deadline_floor_s) for s, d in deadlines.items()}
        self.end_stages = set(end_stages)
        self.recover_q = recover_q    # recovery level (reporting only)
        # 'age': on track while case age <= D_k; 'sojourn': while time in the current
        # stage <= D_k (then D_k are stage-duration deadlines)
        self.mode = mode
        self.t = 0.0
        self.open = 0
        self.off = 0
        self.case = {}           # id -> [arrival, stage, visit_token, off_track]
        self.heap = []           # (crossing time, id, visit_token)
        self.loss = 0.0          # integral of 1 - Q since reset_loss()
        self.q_area = 0.0        # integral of Q since reset_loss()
        self.area_t0 = 0.0
        self.expiries = 0        # expiries caused by a delay (timer won the race)
        self.history = deque()   # (t, Q) breakpoints, kept for the rolling mean
        self.window = 86400.0
        self.hits = {}           # id -> visit token of the stage it was hit in
        self.g = 0.0             # integral of the Loss rate per off-track case
        self.share = {}          # id -> Loss share accrued in closed off-track spells
        self.off_since = {}      # id -> g when its current off-track spell began
        self.expired_at = {}     # id -> g at its expiry

    # ---------------------------------------------------------------- state
    @property
    def q(self):
        return 1.0 - self.off / self.open if self.open else 1.0

    def _integrate_to(self, t):
        dt = t - self.t
        if dt > 0:
            q = self.q
            self.q_area += q * dt
            if self.off:
                self.loss += (1.0 - q) * dt
                self.g += dt / self.open
            self.t = t

    def _mark(self):
        self.history.append((self.t, self.q))
        while len(self.history) > 2 and self.history[1][0] <= self.t - self.window:
            self.history.popleft()

    def advance(self, t):
        """Integrate up to t, processing deadline crossings in time order."""
        while self.heap and self.heap[0][0] <= t:
            tc, cid, token = heapq.heappop(self.heap)
            c = self.case.get(cid)
            if c is None or c[2] != token or c[3]:
                continue
            self._integrate_to(tc)
            c[3] = True
            self.off += 1
            self.off_since[cid] = self.g
            self._mark()
        self._integrate_to(t)

    def _enter(self, cid, stage, t):
        c = self.case[cid]
        self.hits.pop(cid, None)          # a hit case left the stage it was hit in
        if c[3]:
            self.off -= 1
            self._close(cid)
        c[1], c[2] = stage, c[2] + 1
        cross = (c[0] if self.mode == 'age' else t) + self.deadline.get(stage, float('inf'))
        c[3] = t >= cross
        if c[3]:
            self.off += 1
            self.off_since[cid] = self.g
        elif cross < float('inf'):
            heapq.heappush(self.heap, (cross, cid, c[2]))

    # ------------------------------------------------------------ observer
    def on_arrival(self, cid, t):
        self.advance(t)
        self.case[cid] = [t, None, 0, False]
        self.open += 1
        self._mark()

    def on_stage(self, cid, stage, t):
        self.advance(t)
        if stage in self.end_stages:
            return
        self._enter(cid, stage, t)
        self._mark()

    def on_end(self, cid, t):
        self.advance(t)
        c = self.case.pop(cid)
        self.hits.pop(cid, None)
        self.open -= 1
        if c[3]:
            self.off -= 1
            self._close(cid)
        self._mark()

    def on_expiry(self, cid, t):
        self.advance(t)
        self.expiries += 1
        self.expired_at[cid] = self.g

    def _close(self, cid):
        """End the off-track spell of cid (its share accrued so far is kept)."""
        self.share[cid] = self.share.get(cid, 0.0) + self.g - self.off_since.pop(cid)

    def case_loss(self, cid):
        """Loss share of case cid so far (since the start; differences are what matter),
        including the charge after an expiry. Call advance(t) first."""
        v = self.share.get(cid, 0.0)
        if cid in self.off_since:
            v += self.g - self.off_since[cid]
        if cid in self.expired_at:
            v += self.g - self.expired_at[cid]
        return v

    def hit(self, cid):
        """Register that the disruption blocked a step of case cid."""
        if cid in self.case:
            self.hits[cid] = self.case[cid][2]

    def clear_hit(self, cid):
        """The blocked step of case cid has been carried out."""
        self.hits.pop(cid, None)

    # observer calls from the simulator's failure handling
    def on_blocked(self, cid, t):
        self.advance(t)
        self.hit(cid)

    def on_unblocked(self, cid, t):
        self.advance(t)
        self.clear_hit(cid)

    # -------------------------------------------------------------- readout
    def recovered(self, t, disruption_end):
        """Disruption over, every blocked step carried out, and the 24 h rolling mean
        of Q back at recover_q or above."""
        if t < disruption_end or self.hits:
            return False
        return self.rolling_q(t) >= self.recover_q
    def reset_loss(self, t):
        """Start a fresh Loss / Q integral at t (e.g. at the disruption onset)."""
        self.advance(t)
        self.loss = 0.0
        self.q_area = 0.0
        self.area_t0 = t
        self.expiries = 0

    def rolling_q(self, t):
        """Time-weighted mean of Q over the last `window` seconds before t (over the
        available history if it is shorter)."""
        self.advance(t)
        self._mark()
        lo = t - self.window
        area, start, prev_t, prev_q = 0.0, None, None, None
        for tp, qp in self.history:
            if prev_q is not None:
                a, b = max(prev_t, lo), min(tp, t)
                if b > a:
                    area += prev_q * (b - a)
                    start = a if start is None else start
            prev_t, prev_q = tp, qp
        if start is None or t <= start:
            return self.q
        return area / (t - start)
