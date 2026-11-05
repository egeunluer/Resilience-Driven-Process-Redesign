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

Random streams (common random numbers): every source of randomness has its own
stream, derived from the seed -- arrivals, worker attendance, the escalation team's
attendance, and per case: one stream per stage visit for routing and waits, one per
stage visit for work durations, and one for the outcomes of its recovery actions. A
recovery action therefore changes neither other cases' draws nor the case's own draws
in later stage visits, and two runs of the same seed that differ only in actions stay
comparable case by case (loan_env's do-nothing twin relies on this).

Disruptions (Phase 5): with a loan_model.disruptions.DisruptionLayer attached
(`sim.attach_disruption(layer)`), a work session or stage transition it blocks makes
the case fail. The case then waits for a decision: `decision_policy(sim, decision)`
answers at once if set; otherwise the Decision goes to `sim.pending` and the
controller (the gym environment) answers it with `decision.event.succeed(action)`.

Recovery actions (Phase 6, parameters in the disruption config's "recovery" block):
  do_nothing  wait until the service is back (or the normal loop runs its course)
  retry       try again after a short working-time delay (at most
              retry.max_per_case times per case, then masked)
  reassign    missing documents: a caseworker chases the documents (no portal
              delay) and the case's calls go to the front of the queue
  substitute  alternative channel (online-only offer / written contact); side
              effects: later rework or a slower customer response
  escalate    a backup worker handles the step manually (see escalation_team)
`allowed_actions(decision)` gives the mask (escalate only while a backup worker
qualified for the step is on shift and free, retry only while the case has retries
left).
"""
import json
import math
from dataclasses import dataclass, field
from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta

import numpy as np
import simpy

from loan_model.calendar_time import WorkingClock


class StageModel:
    def __init__(self, path_or_dict, pool=None):
        """pool: optional list of worker ids to keep (a constrained resource pool);
        default: every worker in the model."""
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
        self.workers = m['workers'] if pool is None else {w: m['workers'][w] for w in pool}
        self.pool = pool
        self.qualified = defaultdict(list)
        self.median_pt = {}
        for w, spec in self.workers.items():
            for item, d in spec['sessions'].items():
                self.qualified[item].append(w)
                self.median_pt[(w, item)] = d['session_duration'][len(d['session_duration']) // 2]


def top_k_pool(model_dict, k):
    """Constrained pool: for every work item, the k workers with the most sessions
    for it in the log (union over work items)."""
    pool = set()
    for item in model_dict['work_items']:
        ranked = sorted((w for w, spec in model_dict['workers'].items() if item in spec['sessions']),
                        key=lambda w: -model_dict['workers'][w]['sessions'][item]['n'])
        pool.update(ranked[:k])
    return sorted(pool)


def escalation_team(model_dict, pool, work_items, per_item):
    """Backup workers for escalations: for every work item an escalation does by hand,
    the per_item workers with the most sessions for it in the log among those outside
    the constrained pool (the next ranks after the pool's). They are real workers of the
    log -- qualification and attendance are mined -- who only take escalations.
    Returns {work item: [workers]}."""
    if pool is None:
        raise ValueError('the escalation team is drawn from outside a constrained pool')
    pool = set(pool)
    team = {}
    for item in dict.fromkeys(work_items):
        ranked = sorted((w for w, spec in model_dict['workers'].items()
                         if item in spec['sessions'] and w not in pool),
                        key=lambda w: (-model_dict['workers'][w]['sessions'][item]['n'], w))
        team[item] = ranked[:per_item]
    return team


def sample_quantiles(q, rng):
    """Inverse-CDF sample from an array of equally spaced quantiles (0..100 %)."""
    x = rng.random() * (len(q) - 1)
    i = int(x)
    if i >= len(q) - 1:
        return float(q[-1])
    return float(q[i] + (q[i + 1] - q[i]) * (x - i))


def sample_choice(shares, rng):
    """Draw a key from {key: probability} (probabilities need not sum exactly to 1), by
    inverse CDF from one uniform draw."""
    u = rng.random() * sum(shares.values())
    acc = 0.0
    for k, p in shares.items():
        acc += p
        if u < acc:
            return k
    return k


class LazyStream:
    """A random stream created on its first draw (most per-case streams are never or
    rarely used, and creating one costs ~10 us)."""
    __slots__ = ('key', 'gen')

    def __init__(self, key):
        self.key, self.gen = key, None

    def __getattr__(self, name):
        if self.gen is None:
            self.gen = np.random.default_rng(np.random.SeedSequence(self.key))
        return getattr(self.gen, name)


@dataclass
class Decision:
    """A case blocked by a disruption, waiting for the agent's recovery action."""
    cid: int
    failure_type: str
    stage: str
    attempts: int
    t: float
    cause: str = None        # failure that blocked the case (differs for SLA-risk alerts)
    event: object = None
    cancelled: bool = False


class Simulator:
    def __init__(self, model, seed=None, start=datetime(2016, 3, 7), record=True, observers=()):
        self.sm = model if isinstance(model, StageModel) else StageModel(model)
        self.seed = int(np.random.SeedSequence().entropy) if seed is None else int(seed)
        self.arr_rng = self._stream(1)          # arrivals
        self.att_rng = self._stream(2)          # worker attendance
        self.esc_rng = self._stream(3)          # escalation team's attendance
        self.start = start
        self.env = simpy.Environment()
        self.record = record
        # observers (e.g. loan_model.metrics.QTracker) get on_arrival / on_stage /
        # on_end / on_expiry calls with the simulation time
        self.observers = list(observers)
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
        # sorted, not a set: _serve breaks ties between equally old requests by this
        # order, and set order of strings changes with the per-process hash seed
        self.worker_items = {w: sorted(spec['sessions']) for w, spec in self.sm.workers.items()}
        # bookkeeping (event log)
        self.cases = {}                             # id -> dict
        self.sessions = []                          # (case, item, worker, t_request, t0, t1)
        self.shift_seconds = Counter()              # worker -> seconds on shift
        self.next_case_id = 0
        # disruption / decisions (Phase 5)
        self.disruption = None
        self.recovery = None
        self.esc_team = {}                          # work item -> backup workers (escalation)
        self.esc_on_shift = set()
        self.esc_busy = set()
        self.action_cost = 0.0
        self.decision_policy = None
        self.pending = deque()                      # Decisions waiting for the controller
        self.decision_ready = None                  # event the controller waits on (succeeded on a new Decision)
        self.decisions = []                         # (t, case, failure_type, stage, attempts, action)
        self.cohort = set()                         # demand spike: cases that arrived during it
        self.arrival_log = deque(maxlen=24)         # (hour start, arrivals) of the last 24 hours
        self.env.process(self._calendar())
        self.env.process(self._arrivals())

    # ------------------------------------------------------------- time helpers
    def local(self, t):
        return self.start + timedelta(seconds=t)

    def _stream(self, *key):
        return LazyStream([self.seed, *key])

    def _visit_streams(self, cid, v):
        """Fresh streams for the case's v-th stage visit (0: arrival): routing/waits
        and work durations."""
        c = self.cases[cid]
        c['rng'] = self._stream(4, cid, v)
        c['srng'] = self._stream(5, cid, v)

    def _seconds_until(self, dt):
        return (dt - self.local(self.env.now)).total_seconds()

    def _wait_s(self, tr, field, rng):
        """Seconds of a mined wait: in working time when the model has the "_w"
        quantiles (waits that end in a session), else in wall-clock time."""
        qw = tr.get(field + '_w')
        if qw:
            now = self.start_s + self.env.now
            end = self.clock.from_working(self.clock.to_working(now) + sample_quantiles(qw, rng))
            return max(0.0, end - now)
        return sample_quantiles(tr[field], rng)

    def _pause(self, cid, seconds):
        """A wait within a stage visit (mostly the customer's). For a case of the demand
        spike's cohort it can be shortened while it runs (_speed_up: reminder or call);
        other cases get a plain timeout, so the other scenarios are unchanged."""
        if cid not in self.cohort:
            yield self.env.timeout(seconds)
            return
        c = self.cases[cid]
        wf = c.get('wait_factor')
        if wf and wf[0] == len(c['stages']):
            seconds *= wf[1]
        end = self.env.now + seconds
        while True:
            wake = c['wake'] = self.env.event()
            timer = self.env.timeout(max(0.0, end - self.env.now))
            res = yield timer | wake
            if wake not in res:
                break
            end = self.env.now + (end - self.env.now) * wake.value
        c.pop('wake', None)

    def _speed_up(self, cid, token, factor):
        """Multiply the remaining waits of the case's current stage visit by factor."""
        c = self.cases[cid]
        if c['end'] is not None or len(c['stages']) != token:
            return
        old = c.get('wait_factor')
        c['wait_factor'] = (token, factor * (old[1] if old and old[0] == token else 1.0))
        wake = c.get('wake')
        if wake is not None and not wake.triggered:
            wake.succeed(factor)

    # ---------------------------------------------------------------- calendar
    def _calendar(self):
        """Each day: draw who is present, open the shift at hour_min, close at hour_max."""
        day0 = self.local(self.env.now).replace(hour=0, minute=0, second=0, microsecond=0)
        while True:
            wd = day0.weekday()
            if wd in self.cal_days:
                present = [w for w, spec in self.sm.workers.items()
                           if self.att_rng.random() < min(1.0, spec['active_day_share'] * self.day_factor[wd])]
                team = sorted({w for ws in self.esc_team.values() for w in ws})
                esc_present = [w for w in team if self.esc_rng.random() < min(
                    1.0, self.sm.m['workers'][w]['active_day_share'] * self.day_factor[wd])]
                yield self.env.timeout(max(0.0, self._seconds_until(day0.replace(hour=self.h_min))))
                for w in present:
                    self.on_shift.add(w)
                    self.shift_seconds[w] += (self.h_max - self.h_min) * 3600
                    if w not in self.busy:
                        self._serve(w)
                self.esc_on_shift.update(esc_present)
                yield self.env.timeout(max(0.0, self._seconds_until(day0.replace(hour=self.h_max))))
                self.on_shift.clear()
                self.esc_on_shift.clear()
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
        elif self._has_priority(case_id):
            self.queues[item].appendleft((-1.0, ev, case_id))   # reassign: front of the queue
        else:
            self.queues[item].append((self.env.now, ev, case_id))
        return ev

    def _has_priority(self, cid):
        c = self.cases.get(cid)
        return bool(c) and c.get('priority_visit') == (c['stages'][-1][0], len(c['stages']))

    def release_worker(self, w):
        self.busy.discard(w)
        if w in self.on_shift:
            self._serve(w)

    def session_duration(self, w, item, rng):
        spec = self.sm.workers[w]['sessions'].get(item)
        q = spec['session_duration'] if spec else self.sm.m['work_items'][item]['session_duration']
        return sample_quantiles(q, rng)

    # ---------------------------------------------------------------- arrivals
    def _arrivals(self):
        rates = self.sm.m['arrivals']['rates_per_hour_of_week']
        while True:
            now = self.local(self.env.now)
            slot = now.weekday() * 24 + now.hour
            # demand spike: the rate of the hours that start during it (a factor of 1
            # leaves the draws, and so every other scenario, unchanged)
            factor = self.disruption.arrival_factor() if self.disruption is not None else 1.0
            n = self.arr_rng.poisson(rates[slot] * factor)
            self.arrival_log.append((self.env.now, n))
            offsets = np.sort(self.arr_rng.uniform(0, 3600, n))
            t_hour = self.env.now
            for off in offsets:
                yield self.env.timeout(t_hour + off - self.env.now)
                self._new_case()
            yield self.env.timeout(t_hour + 3600 - self.env.now)

    def _new_case(self):
        cid = self.next_case_id
        self.next_case_id += 1
        self.cases[cid] = c = {'arrival': self.env.now, 'stages': [], 'end': None, 'outcome': None, 'offers': 0,
                               'arng': self._stream(6, cid)}      # outcomes of its recovery actions
        self._visit_streams(cid, 0)
        c['flags'] = {f for f, p in self.sm.m['case_flags'].items() if c['rng'].random() < p}
        if self.disruption is not None and self.disruption.arrival_factor() != 1.0:
            self.cohort.add(cid)
        for o in self.observers:
            o.on_arrival(cid, self.env.now)
        self.env.process(self._case(cid))

    # -------------------------------------------------------------------- case
    def _case(self, cid):
        c = self.cases[cid]
        stage = self.sm.start_stage
        visits = Counter()
        while stage not in self.sm.end_stages:
            visits[stage] += 1
            c['stages'].append((stage, self.env.now))
            self._visit_streams(cid, len(c['stages']))
            for o in self.observers:
                o.on_stage(cid, stage, self.env.now)
            stage = yield from self._visit(cid, stage, visits[stage])
        c['stages'].append((stage, self.env.now))
        c['end'] = self.env.now
        c['outcome'] = stage
        for o in self.observers:
            o.on_end(cid, self.env.now)

    def _visit(self, cid, stage, v):
        """One stage visit; returns the next stage."""
        st = self.sm.stages[stage]
        c = self.cases[cid]
        rng = c['rng']
        visit_flags = set()
        for item, spec in st['optional_work_items'].items():
            if rng.random() < spec['p_per_visit']:
                visit_flags.add(spec['flag'])
                yield from self._session(cid, item)
        flagged = [f for f in visit_flags if f in st['exits_by_flag']]
        exits = st['exits_by_flag'][flagged[0]] if flagged else st['exits_by_visit'][str(min(v, self.sm.max_bucket))]
        if self.disruption is not None:
            exits = self.disruption.modify_exits(stage, exits)
        nxt = sample_choice(exits, rng)
        tr = st['transitions'][nxt]
        timers = self.sm.timers_by_stage.get(stage, [])
        if timers:
            # the visit races against the timer (e.g. offer expiry): whichever ends first
            tm = timers[0]
            body = self.env.process(self._visit_body(cid, st, tr))
            expiry = None
            if nxt != tm['target']:
                delay = sample_quantiles(tm['delay'], rng)
                expiry = self.env.timeout(delay)
                c['expiry_at'] = self.env.now + delay       # seen by the agent
            if expiry is None:
                yield body
            else:
                res = yield body | expiry
                c.pop('expiry_at', None)
                if body not in res:
                    body.interrupt('timer')
                    c.setdefault('timers', []).append(tm['name'])
                    for o in self.observers:
                        o.on_expiry(cid, self.env.now)
                    nxt = tm['target']
        else:
            yield from self._visit_body(cid, st, tr)
        if self.disruption is not None:
            # a blocked transition (e.g. offer dispatch down) holds the case in this stage
            if self.disruption.blocks_transition(stage, nxt):
                yield from self._blocked(cid, stage, lambda: self.disruption.blocks_transition(stage, nxt))
            ftype = self.disruption.surge_decision(stage, nxt)
            if ftype:
                nxt = yield from self._documents_decision(cid, ftype, stage, nxt, exits)
        if stage == 'complete' and nxt not in self.sm.end_stages:
            c['offers'] += int(sample_choice(self.sm.m['offers']['per_case'], rng))
        return nxt

    def _visit_body(self, cid, st, tr):
        try:
            extra = self.cases[cid].pop('extra_wait', None)
            if extra:
                # documents surge side effect: the customer's upload is held up
                yield self.env.timeout(extra)
            rework = self.cases[cid].pop('rework_delay', None)
            if rework:
                # substitute side effect: the step has to be redone in this stage
                yield self.env.timeout(rework)
                if st['work_item']:
                    yield from self._session(cid, st['work_item'])
            rng = self.cases[cid]['rng']
            k = int(sample_choice(tr['n_sessions'], rng))
            if k == 0:
                q = tr.get('visit_duration_no_session') or tr['visit_duration']
                yield from self._pause(cid, sample_quantiles(q, rng))
                return
            yield from self._pause(cid, self._wait_s(tr, 'initial_delay', rng))
            for i in range(k):
                yield from self._session(cid, st['work_item'])
                if i < k - 1:
                    yield from self._pause(cid, self._wait_s(tr, 'gap', rng))
            yield from self._pause(cid, self._wait_s(tr, 'tail', rng))
        except simpy.Interrupt:
            return

    def _session(self, cid, item):
        if self.disruption is not None and self.disruption.blocks_session(item):
            stage = self.cases[cid]['stages'][-1][0]
            bypassed = yield from self._blocked(cid, stage, lambda: self.disruption.blocks_session(item))
            if bypassed:
                return          # done another way (written contact / senior team)
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
            yield self.env.timeout(self.session_duration(w, item, self.cases[cid]['srng']))
        finally:
            self.release_worker(w)
            if self.record:
                self.sessions.append((cid, item, w, t_req, t0, self.env.now))

    # ------------------------------------------------------- failures/decisions
    BYPASS = 'bypass'       # _recover result: the blocked step was done another way

    def attach_disruption(self, layer):
        self.disruption = layer
        self.recovery = layer.cfg.get('recovery')
        if self.recovery:
            spec = self.recovery['escalate']
            self.esc_item = spec['work_items']
            self.esc_team = escalation_team(self.sm.m, self.sm.pool, list(self.esc_item.values()),
                                            spec['team_per_item'])
        if layer.mechanism == 'arrival_surge':
            self.env.process(self._spike_watch())

    def _notify(self, name, *args):
        for o in self.observers:
            f = getattr(o, name, None)
            if f is not None:
                f(*args)

    def allowed_actions(self, d):
        """Action mask for decision d: the failure's allowed actions, escalate only
        while a backup worker for the step is free."""
        if not self.recovery:
            return ['do_nothing']
        acts = list(self.recovery['allowed'][d.cause or d.failure_type])
        if 'escalate' in acts and not self.escalation_free(d.cause or d.failure_type):
            acts.remove('escalate')
        cap = self.recovery['retry'].get('max_per_case')
        if 'retry' in acts and cap is not None and self.cases[d.cid].get('retries', 0) >= cap:
            acts.remove('retry')
        return acts

    def _decide(self, cid, ftype, stage, attempts, cause=None):
        """Ask for a recovery action; returns it."""
        d = Decision(cid, ftype, stage, attempts, self.env.now, cause=cause or ftype)
        if self.decision_policy is not None:
            action = self.decision_policy(self, d)
            if action not in self.allowed_actions(d):
                raise ValueError(f'action {action!r} not allowed for {d}')
        else:
            d.event = self.env.event()
            self.pending.append(d)
            if self.decision_ready is not None and not self.decision_ready.triggered:
                self.decision_ready.succeed()
            try:
                action = yield d.event
            except simpy.Interrupt:
                d.cancelled = True
                if d in self.pending:
                    self.pending.remove(d)
                raise
        if self.recovery:
            self.action_cost += self.recovery['kappa'][action]
        if action == 'retry':
            self.cases[cid]['retries'] = self.cases[cid].get('retries', 0) + 1
        self.decisions.append((d.t, cid, ftype, stage, attempts, action))
        return action

    def _working_timeout(self, hours):
        now = self.start_s + self.env.now
        end = self.clock.from_working(self.clock.to_working(now) + hours * 3600)
        return self.env.timeout(max(0.0, end - now))

    def escalation_free(self, cause):
        """Backup workers for the step of this failure, on shift and free (sorted)."""
        return [w for w in self.esc_team.get(self.esc_item.get(cause), [])
                if w in self.esc_on_shift and w not in self.esc_busy]

    def escalation_size(self, cause):
        return len(self.esc_team.get(self.esc_item.get(cause), []))

    def _escalate(self, cid, cause):
        """A free backup worker does the step by hand (handling time assumed: a manual
        workaround, slower than the logged system-supported session); True if it
        succeeds. Taken at once, so the worker is busy before the next decision."""
        spec = self.recovery['escalate']
        arng = self.cases[cid]['arng']
        hours, ok = float(arng.uniform(*spec['handling_working_h']['uniform'])), arng.random() < spec['success_p']
        w = self.escalation_free(cause)[0]
        self.esc_busy.add(w)
        try:
            yield self._working_timeout(hours)
        finally:
            self.esc_busy.discard(w)
        return ok

    def _blocked(self, cid, stage, still_blocked):
        """The case cannot go on while still_blocked() returns a failure type: ask for
        decisions until it can. Returns True if the step was done another way."""
        self._notify('on_blocked', cid, self.env.now)
        ctx = {'alerted': False}
        attempts = 0
        try:
            cause = ftype = still_blocked()
            while ftype:
                attempts += 1
                action = yield from self._decide(cid, ftype, stage, attempts, cause)
                ftype = yield from self._recover(cid, action, cause, stage, still_blocked, ctx)
                if ftype == self.BYPASS:
                    return True
            return False
        finally:
            self._notify('on_unblocked', cid, self.env.now)

    def _recover(self, cid, action, cause, stage, still_blocked, ctx):
        """Carry out a recovery action for a blocked step; returns the failure type
        of the next decision, None when the case can go on, or BYPASS."""
        d, rc = self.disruption, self.recovery
        if action == 'do_nothing':
            if still_blocked() is None or d.restored.triggered:
                return still_blocked()
            if not ctx['alerted']:
                delay = d.alert_delay(self.cases[cid]['arrival'], stage, self.env.now)
                if delay is not None:
                    yield d.restored | self.env.timeout(delay)
                    if not d.restored.triggered:
                        ctx['alerted'] = True       # D6: slack is running out, decide again
                        return d.alert_type
                    return still_blocked()
            yield d.restored
            return still_blocked()
        if action == 'retry':
            yield self._working_timeout(rc['retry']['delay_working_h'])
            return still_blocked()
        if action == 'substitute':
            spec, arng = rc['substitute'][cause], self.cases[cid]['arng']
            # all draws first, so the stream's use does not depend on the branch taken
            extra = 3600 * float(arng.uniform(*spec['extra_delay_h']['uniform'])) if 'extra_delay_h' in spec else 0.0
            rework = arng.random() < spec.get('rework_p', 0.0)
            rework_s = 3600 * float(arng.uniform(*spec['rework_delay_h']['uniform'])) if 'rework_delay_h' in spec else 0.0
            if extra:
                yield self.env.timeout(extra)
            if rework:
                self.cases[cid]['rework_delay'] = rework_s
            return self.BYPASS
        if action == 'escalate':
            ok = yield from self._escalate(cid, cause)
            return self.BYPASS if ok else still_blocked()
        raise ValueError(f'action {action!r} does not apply to a blocked step ({cause})')

    def _spike_watch(self):
        """Demand spike trigger: during working hours, a case of the spike's cohort in a
        customer-facing stage that is still on track but within the alert slack of its
        stage deadline gets one decision per stage visit."""
        d = self.disruption
        trig = d.spec['trigger']
        stages, slack_s = set(trig['stages']), trig['slack_h'] * 3600
        yield self.env.timeout(max(0.0, d.onset - self.env.now))
        while True:
            if self.on_shift:
                now = self.env.now
                for cid in sorted(self.cohort):
                    c = self.cases[cid]
                    if c['end'] is not None:
                        self.cohort.discard(cid)
                        continue
                    stage, token = c['stages'][-1][0], len(c['stages'])
                    if stage not in stages or c.get('alerted') == token:
                        continue
                    if 0.0 < c['arrival'] + d.deadlines[stage] - now <= slack_s:
                        c['alerted'] = token
                        self.env.process(self._spike_decision(cid, stage, token))
            yield self.env.timeout(3600)

    def _spike_decision(self, cid, stage, token):
        action = yield from self._decide(cid, self.disruption.failure_type, stage, 1)
        rc, ft, arng = self.recovery, self.disruption.failure_type, self.cases[cid]['arng']
        if action == 'reassign':
            self.cases[cid]['priority_visit'] = (stage, token)      # its calls go first
        elif action == 'substitute':
            self._speed_up(cid, token, float(arng.uniform(*rc['substitute'][ft]['wait_factor']['uniform'])))
        elif action == 'escalate':
            factor = float(arng.uniform(*rc['escalate']['chase_wait_factor']['uniform']))
            if (yield from self._escalate(cid, ft)):
                self._speed_up(cid, token, factor)

    def _documents_decision(self, cid, ftype, stage, nxt, exits):
        """A case routed to the incomplete loop during a documents surge: the action
        may avoid the loop (re-fetch, senior team) or speed it up (reassign: no
        portal delay, calls first). Left alone, the loop also waits for the faulty
        portal (the level's upload_delay_h)."""
        action = yield from self._decide(cid, ftype, stage, 1)
        rc, arng = self.recovery, self.cases[cid]['arng']
        delay = self.disruption.params.get('upload_delay_h')
        # draws first, in a fixed order, so the stream's use does not depend on the branch
        upload_s = 3600 * float(arng.uniform(*delay['uniform'])) if delay else 0.0
        u_alt = arng.random()
        avoided = False
        if action == 'retry':
            yield self._working_timeout(rc['retry']['delay_working_h'])
            avoided = arng.random() < rc['retry']['documents_success_p']
        elif action == 'escalate':
            avoided = yield from self._escalate(cid, ftype)
        elif action == 'reassign':
            c = self.cases[cid]
            c['priority_visit'] = (nxt, len(c['stages']) + 1)   # the coming incomplete visit
        if not avoided and action != 'reassign' and upload_s:
            self.cases[cid]['extra_wait'] = upload_s
        if avoided:
            others = {k: v for k, v in exits.items() if k != nxt}
            if others:
                keys = list(others)
                p = np.cumsum([others[k] for k in keys])
                nxt = keys[min(int(np.searchsorted(p, u_alt * p[-1], side='right')), len(keys) - 1)]
        return nxt

    # --------------------------------------------------------------------- run
    def run(self, days):
        self.env.run(until=days * 86400)
        return self
