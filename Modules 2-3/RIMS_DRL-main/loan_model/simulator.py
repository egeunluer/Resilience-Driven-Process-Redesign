"""
Generic stage-model simulator (Phase 3). Reads a `<log>_stage_model.json` written by
loan_model/mine_stage_model.py and simulates it with SimPy; semantics in
loan_model/STAGE_MODEL_SCHEMA.md. Nothing here is specific to BPIC2017.

Resources: every human worker in the model is a capacity-1 server qualified for the
work items it did at least `min_worker_sessions` times in the log. Each day a worker
is present with probability active_day_share x (weekday's session share relative to
the Mon-Fri mean), on shift from calendar.hour_min to hour_max local time.

Fixed dispatcher (the agent never allocates resources): a session request goes to the
fastest free qualified worker (lowest median session duration for that work item);
if none is free it waits in a FIFO queue per work item; a worker who becomes free
(or comes on shift) takes the oldest waiting request it is qualified for.

Time: simulation seconds from `start` (a naive local datetime, default a Monday 00:00).
"""
import json
import math
from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta

import numpy as np
import simpy

from loan_model.calendar_time import WorkingClock


class StageModel:
    def __init__(self, path_or_dict):
        if isinstance(path_or_dict, dict):
            self.m = path_or_dict
        else:
            with open(path_or_dict) as f:
                self.m = json.load(f)
        m = self.m
        self.stages = m['stages']
        self.end_stages = set(m['end_stages'])
        self.start_stage = m['start_stage']
        self.timers_by_stage = defaultdict(list)
        for name, tm in m['timers'].items():
            if tm['delay']:
                self.timers_by_stage[tm['stage']].append(dict(tm, name=name))
        self.buckets = sorted(int(b) for s in self.stages.values() for b in s['exits_by_visit'])
        self.max_bucket = max(self.buckets)
        # qualified workers and their median session duration per work item
        self.workers = m['workers']
        self.qualified = defaultdict(list)
        self.median_pt = {}
        for w, spec in self.workers.items():
            for item, d in spec['sessions'].items():
                self.qualified[item].append(w)
                self.median_pt[(w, item)] = d['session_duration'][len(d['session_duration']) // 2]


def sample_quantiles(q, rng):
    """Inverse-CDF sample from an array of equally spaced quantiles (0..100 %)."""
    x = rng.random() * (len(q) - 1)
    i = int(x)
    if i >= len(q) - 1:
        return float(q[-1])
    return float(q[i] + (q[i + 1] - q[i]) * (x - i))


def sample_choice(shares, rng):
    """Draw a key from {key: probability} (probabilities need not sum exactly to 1)."""
    keys = list(shares)
    p = np.array([shares[k] for k in keys], dtype=float)
    return keys[rng.choice(len(keys), p=p / p.sum())]


class Simulator:
    def __init__(self, model, seed=None, start=datetime(2016, 3, 7), record=True):
        self.sm = model if isinstance(model, StageModel) else StageModel(model)
        self.rng = np.random.default_rng(seed)
        self.start = start
        self.env = simpy.Environment()
        self.record = record
        cal = self.sm.m['calendar']
        self.cal_days = set(cal['days'])
        self.h_min, self.h_max = cal['hour_min'], cal['hour_max']
        share = cal['session_share_by_weekday']
        wk_mean = np.mean(share[:5])
        self.day_factor = [s / wk_mean for s in share]
        self.clock = WorkingClock(cal['days'], cal['hour_min'], cal['hour_max'])
        self.start_s = WorkingClock.seconds(start)
        # dispatcher state
        self.on_shift = set()
        self.busy = set()
        self.queues = defaultdict(deque)          # work item -> deque of (t_request, event, case_id)
        self.worker_items = {w: set(spec['sessions']) for w, spec in self.sm.workers.items()}
        # bookkeeping (event log)
        self.cases = {}                             # id -> dict
        self.sessions = []                          # (case, item, worker, t_request, t0, t1)
        self.shift_seconds = Counter()              # worker -> seconds on shift
        self.next_case_id = 0
        self.env.process(self._calendar())
        self.env.process(self._arrivals())

    # ------------------------------------------------------------- time helpers
    def local(self, t):
        return self.start + timedelta(seconds=t)

    def _seconds_until(self, dt):
        return (dt - self.local(self.env.now)).total_seconds()

    def _wait(self, tr, field):
        """Timeout for a mined wait: in working time when the model has the "_w"
        quantiles (waits that end in a session), else in wall-clock time."""
        qw = tr.get(field + '_w')
        if qw:
            now = self.start_s + self.env.now
            end = self.clock.from_working(self.clock.to_working(now) + sample_quantiles(qw, self.rng))
            return self.env.timeout(max(0.0, end - now))
        return self.env.timeout(sample_quantiles(tr[field], self.rng))

    # ---------------------------------------------------------------- calendar
    def _calendar(self):
        """Each day: draw who is present, open the shift at hour_min, close at hour_max."""
        day0 = self.local(self.env.now).replace(hour=0, minute=0, second=0, microsecond=0)
        while True:
            wd = day0.weekday()
            if wd in self.cal_days:
                present = [w for w, spec in self.sm.workers.items()
                           if self.rng.random() < min(1.0, spec['active_day_share'] * self.day_factor[wd])]
                yield self.env.timeout(max(0.0, self._seconds_until(day0.replace(hour=self.h_min))))
                for w in present:
                    self.on_shift.add(w)
                    self.shift_seconds[w] += (self.h_max - self.h_min) * 3600
                    if w not in self.busy:
                        self._serve(w)
                yield self.env.timeout(max(0.0, self._seconds_until(day0.replace(hour=self.h_max))))
                self.on_shift.clear()
            day0 += timedelta(days=1)
            yield self.env.timeout(max(0.0, self._seconds_until(day0)))

    # -------------------------------------------------------------- dispatcher
    def _serve(self, w):
        """Free worker w takes the oldest waiting request it is qualified for."""
        best = None
        for item in self.worker_items[w]:
            q = self.queues[item]
            if q and (best is None or q[0][0] < best[1][0]):
                best = (item, q[0])
        if best is None:
            return
        self.queues[best[0]].popleft()
        self.busy.add(w)
        best[1][1].succeed(w)

    def request_worker(self, item, case_id):
        """Event that fires with the assigned worker id."""
        ev = self.env.event()
        free = [w for w in self.sm.qualified[item] if w in self.on_shift and w not in self.busy]
        if free:
            w = min(free, key=lambda x: self.sm.median_pt[(x, item)])
            self.busy.add(w)
            ev.succeed(w)
        else:
            self.queues[item].append((self.env.now, ev, case_id))
        return ev

    def release_worker(self, w):
        self.busy.discard(w)
        if w in self.on_shift:
            self._serve(w)

    def session_duration(self, w, item):
        spec = self.sm.workers[w]['sessions'].get(item)
        q = spec['session_duration'] if spec else self.sm.m['work_items'][item]['session_duration']
        return sample_quantiles(q, self.rng)

    # ---------------------------------------------------------------- arrivals
    def _arrivals(self):
        rates = self.sm.m['arrivals']['rates_per_hour_of_week']
        while True:
            now = self.local(self.env.now)
            slot = now.weekday() * 24 + now.hour
            n = self.rng.poisson(rates[slot])
            offsets = np.sort(self.rng.uniform(0, 3600, n))
            t_hour = self.env.now
            for off in offsets:
                yield self.env.timeout(t_hour + off - self.env.now)
                self._new_case()
            yield self.env.timeout(t_hour + 3600 - self.env.now)

    def _new_case(self):
        cid = self.next_case_id
        self.next_case_id += 1
        flags = {f for f, p in self.sm.m['case_flags'].items() if self.rng.random() < p}
        self.cases[cid] = {'arrival': self.env.now, 'flags': flags, 'stages': [], 'end': None,
                           'outcome': None, 'offers': 0}
        self.env.process(self._case(cid))

    # -------------------------------------------------------------------- case
    def _case(self, cid):
        c = self.cases[cid]
        stage = self.sm.start_stage
        visits = Counter()
        while stage not in self.sm.end_stages:
            visits[stage] += 1
            c['stages'].append((stage, self.env.now))
            stage = yield from self._visit(cid, stage, visits[stage])
        c['stages'].append((stage, self.env.now))
        c['end'] = self.env.now
        c['outcome'] = stage

    def _visit(self, cid, stage, v):
        """One stage visit; returns the next stage."""
        st = self.sm.stages[stage]
        c = self.cases[cid]
        visit_flags = set()
        for item, spec in st['optional_work_items'].items():
            if self.rng.random() < spec['p_per_visit']:
                visit_flags.add(spec['flag'])
                yield from self._session(cid, item)
        flagged = [f for f in visit_flags if f in st['exits_by_flag']]
        exits = st['exits_by_flag'][flagged[0]] if flagged else st['exits_by_visit'][str(min(v, self.sm.max_bucket))]
        nxt = sample_choice(exits, self.rng)
        tr = st['transitions'][nxt]
        timers = self.sm.timers_by_stage.get(stage, [])
        if timers:
            # the visit races against the timer (e.g. offer expiry): whichever ends first
            tm = timers[0]
            body = self.env.process(self._visit_body(cid, st, tr))
            expiry = self.env.timeout(sample_quantiles(tm['delay'], self.rng)) if nxt != tm['target'] else None
            if expiry is None:
                yield body
            else:
                res = yield body | expiry
                if body not in res:
                    body.interrupt('timer')
                    c.setdefault('timers', []).append(tm['name'])
                    nxt = tm['target']
        else:
            yield from self._visit_body(cid, st, tr)
        if stage == 'complete' and nxt not in self.sm.end_stages:
            c['offers'] += int(sample_choice(self.sm.m['offers']['per_case'], self.rng))
        return nxt

    def _visit_body(self, cid, st, tr):
        try:
            k = int(sample_choice(tr['n_sessions'], self.rng))
            if k == 0:
                q = tr.get('visit_duration_no_session') or tr['visit_duration']
                yield self.env.timeout(sample_quantiles(q, self.rng))
                return
            yield self._wait(tr, 'initial_delay')
            for i in range(k):
                yield from self._session(cid, st['work_item'])
                if i < k - 1:
                    yield self._wait(tr, 'gap')
            yield self._wait(tr, 'tail')
        except simpy.Interrupt:
            return

    def _session(self, cid, item):
        t_req = self.env.now
        req = self.request_worker(item, cid)
        try:
            w = yield req
        except simpy.Interrupt:
            # interrupted while queued: withdraw the request (or hand back a worker
            # assigned in the same instant)
            q = self.queues[item]
            for i, entry in enumerate(q):
                if entry[1] is req:
                    del q[i]
                    break
            else:
                if req.triggered:
                    self.release_worker(req.value)
            raise
        t0 = self.env.now
        try:
            yield self.env.timeout(self.session_duration(w, item))
        finally:
            self.release_worker(w)
            if self.record:
                self.sessions.append((cid, item, w, t_req, t0, self.env.now))

    # --------------------------------------------------------------------- run
    def run(self, days):
        self.env.run(until=days * 86400)
        return self
