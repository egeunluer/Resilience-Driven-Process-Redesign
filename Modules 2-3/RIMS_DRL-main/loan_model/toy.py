"""
Phase 1 proof of concept (drl_resilience_design_decisions.md §16): a tiny triage
world, small enough to train in minutes, that contains all four sources of
difficulty from §4. If a learned policy can't beat the simple baselines here,
DRL isn't justified for the full model either.

World (times in hours, one episode = one outage):
  - An external service (e.g. a credit check) goes down at t=0. Its outage length
    is uncertain: short (1-4h) or long (12-36h) with equal probability. The agent
    sees that it is down and for how long so far, never when it will end.
  - N_CASES cases arrive during the first ARRIVAL_WINDOW hours, each with its own
    deadline (slack drawn per case). Each needs one main-service step.
  - A case that finds the service down fails -> decision event for the agent.
  - Actions (recovery menu):
      do-nothing : wait until the service is restored, then queue for it.
      retry      : try again after RETRY_DELAY; fails again if still down.
      substitute : backup service (1 server, slower, queue shared by all cases);
                   with prob R_SUB the result needs rework later (side-effect).
      escalate   : 1 scarce escalation slot, fast, succeeds with prob Q_ESC;
                   masked while the slot is busy.
  - Coupling: the escalation slot and the backup server are shared by all cases.

Resilience measure: Q(t) = share of open cases still on track (now <= deadline);
Loss = integral of (1 - Q(t)) dt (Q_base = 1: without the outage every case meets
its deadline). Reward between two decisions:
    -(W * dLoss / loss_norm + sum of action costs + MISS_PENALTY * deadlines missed)
with action costs kappa * c_p (c_p = 1).
"""
import math

import numpy as np
import simpy
from gymnasium import Env, spaces

# Actions
DO_NOTHING, RETRY, SUBSTITUTE, ESCALATE = range(4)
ACTION_NAMES = ['do_nothing', 'retry', 'substitute', 'escalate']
KAPPA = {DO_NOTHING: 0.0, RETRY: 0.05, SUBSTITUTE: 0.30, ESCALATE: 0.50}

N_CASES = 30
ARRIVAL_WINDOW = 24.0
SLACK_RANGE = (4.0, 30.0)
OUTAGE_SHORT = (1.0, 4.0)
OUTAGE_LONG = (12.0, 36.0)
P_LONG = 0.5
MAIN_SERVERS, MAIN_TIME = 2, 0.5
RETRY_DELAY = 1.0
BACKUP_SERVERS, BACKUP_TIME, R_SUB, REWORK_DELAY = 1, 2.5, 0.3, 1.0
ESC_SLOTS, ESC_TIME, Q_ESC = 1, 1.5, 0.95
MISS_PENALTY = 1.0
W = 1.0
OBS_DIM = 14


class ToyTriageEnv(Env):
    def __init__(self, loss_norm=1.0, w=W):
        super().__init__()
        self.loss_norm = loss_norm
        self.w = w
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.last_episode_info = {}

    # ------------------------------------------------------------------ world
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.env = simpy.Environment()
        self.main = simpy.Resource(self.env, MAIN_SERVERS)
        self.backup = simpy.Resource(self.env, BACKUP_SERVERS)
        self.esc = simpy.Resource(self.env, ESC_SLOTS)
        long = self.rng.random() < P_LONG
        self.outage_len = float(self.rng.uniform(*(OUTAGE_LONG if long else OUTAGE_SHORT)))
        self.restored = self.env.event()
        self.env.process(self._outage())
        self.cases = {}          # id -> dict(deadline, done, missed, attempts, rework)
        self.pending = []        # decision queue: (case_id, decision_event)
        self.loss = 0.0          # integral of (1 - Q) since t=0
        self._t_loss = 0.0
        self.penalties = 0.0     # deadline misses since last step
        self.cost = 0.0          # action costs, episode total
        self.n_missed = 0
        self.counts = [0, 0, 0, 0]
        # first case arrives at the outage onset, so every episode has a decision
        arrivals = np.sort(np.concatenate([[0.0], self.rng.uniform(0, ARRIVAL_WINDOW, N_CASES - 1)]))
        for i, t in enumerate(arrivals):
            self.env.process(self._case(i, float(t), float(self.rng.uniform(*SLACK_RANGE))))
        self._loss_since_step = 0.0
        self._run_to_decision()
        return self._obs(), {}

    def _outage(self):
        yield self.env.timeout(self.outage_len)
        self.restored.succeed()

    @property
    def outage_active(self):
        return not self.restored.triggered

    def _integrate(self):
        """Advance the Loss integral to env.now; Q(t) is constant between events."""
        now = self.env.now
        dt = now - self._t_loss
        if dt > 0:
            open_ = [c for c in self.cases.values() if not c['done']]
            if open_:
                # off-track status only changes at deadline events, which are SimPy
                # events that integrate up to their own instant before flagging the
                # case, so it is constant over (t_loss, now].
                off_track = sum(1 for c in open_ if c['missed'])
                d = dt * off_track / len(open_)
                self.loss += d
                self._loss_since_step += d
            self._t_loss = now

    def _deadline_watch(self, cid):
        c = self.cases[cid]
        yield self.env.timeout(max(0.0, c['deadline'] - self.env.now))
        if not c['done']:
            self._integrate()
            c['missed'] = True
            self.n_missed += 1
            self.penalties += MISS_PENALTY

    def _case(self, cid, t_arr, slack):
        yield self.env.timeout(t_arr)
        self._integrate()
        self.cases[cid] = dict(deadline=t_arr + slack, done=False, missed=False, attempts=0, rework=False)
        self.env.process(self._deadline_watch(cid))
        while True:
            if not self.outage_active:
                with self.main.request() as req:
                    yield req
                    yield self.env.timeout(MAIN_TIME)
                break
            # service down -> the agent decides
            self.cases[cid]['attempts'] += 1
            ev = self.env.event()
            self.pending.append((cid, ev))
            action = yield ev
            if action == DO_NOTHING:
                yield self.restored
                continue
            if action == RETRY:
                yield self.env.timeout(RETRY_DELAY)
                continue
            if action == ESCALATE:
                with self.esc.request() as req:
                    yield req
                    yield self.env.timeout(ESC_TIME)
                if self.rng.random() < Q_ESC:
                    break
                continue
            if action == SUBSTITUTE:
                with self.backup.request() as req:
                    yield req
                    yield self.env.timeout(BACKUP_TIME)
                if self.rng.random() < R_SUB and not self.cases[cid]['rework']:
                    # side-effect: backup result is rejected later, case needs the
                    # main service after all
                    self.cases[cid]['rework'] = True
                    yield self.env.timeout(REWORK_DELAY)
                    continue
                break
        self._integrate()
        self.cases[cid]['done'] = True

    def _run_to_decision(self):
        while not self.pending and self.env.peek() < math.inf:
            self.env.step()
            self._integrate()
        return bool(self.pending)

    # -------------------------------------------------------------- interface
    def action_masks(self):
        return [True, True, True, self.esc.count < ESC_SLOTS]

    def _obs(self):
        now = self.env.now
        if self.pending:
            c = self.cases[self.pending[0][0]]
            slack = (c['deadline'] - now) / SLACK_RANGE[1]
            att, rew = c['attempts'] / 5, float(c['rework'])
        else:
            slack, att, rew = 0.0, 0.0, 0.0
        open_ = [c for c in self.cases.values() if not c['done']]
        q = 1.0 - (sum(c['missed'] for c in open_) / len(open_) if open_ else 0.0)
        o = [float(self.outage_active), min(now, OUTAGE_LONG[1]) / OUTAGE_LONG[1] if self.outage_active else 0.0,
             float(self.esc.count < ESC_SLOTS), float(self.backup.count) / BACKUP_SERVERS,
             min(len(self.backup.queue), 10) / 10, min(len(self.pending), N_CASES) / N_CASES,
             len(open_) / N_CASES, q, slack, att, rew,
             # system urgency: other open cases close to their deadline, and how much
             # of the arrival window is still to come (competition for the slot later)
             sum(1 for c in open_ if 0 <= c['deadline'] - now < 4) / N_CASES,
             sum(1 for c in open_ if 4 <= c['deadline'] - now < 10) / N_CASES,
             max(0.0, 1 - now / ARRIVAL_WINDOW)]
        return np.clip(np.array(o, dtype=np.float32), -1.0, 1.0)

    def current_case(self):
        """(deadline - now, attempts) of the case the next decision is about."""
        c = self.cases[self.pending[0][0]]
        return c['deadline'] - self.env.now, c['attempts']

    def step(self, action):
        action = int(action)
        cid, ev = self.pending.pop(0)
        self.counts[action] += 1
        self.cost += KAPPA[action]
        ev.succeed(action)
        step_cost = KAPPA[action]
        done = not self._run_to_decision()
        r = -(self.w * self._loss_since_step / self.loss_norm + step_cost + self.penalties)
        self._loss_since_step = 0.0
        self.penalties = 0.0
        info = {}
        if done:
            self.last_episode_info = {
                'loss': self.loss, 'loss_ratio': self.loss / self.loss_norm, 'missed': self.n_missed,
                'action_cost': self.cost, 'outage_h': self.outage_len,
                **{f'n_{n}': k for n, k in zip(ACTION_NAMES, self.counts)},
            }
            info['resilience'] = {'shock': 'TOY_OUTAGE', **self.last_episode_info}
        return self._obs(), r, done, False, info


# ------------------------------------------------------------------ baselines
def _expected_remaining_outage(elapsed):
    """E[outage end - now | outage still on after `elapsed` h], from the known mixture."""
    parts = []
    for p, (a, b) in ((1 - P_LONG, OUTAGE_SHORT), (P_LONG, OUTAGE_LONG)):
        lo = max(a, elapsed)
        if lo >= b:
            continue
        mass = p * (b - lo) / (b - a)
        parts.append((mass, (lo + b) / 2 - elapsed))
    tot = sum(m for m, _ in parts)
    return sum(m * e for m, e in parts) / tot if tot else 0.0


def policy_greedy(env):
    """Myopic: lowest immediate expected cost (action cost + deadline-miss risk for
    this case alone), ignoring the scarce slot's opportunity cost and rework."""
    slack, _ = env.current_case()
    now = env.env.now
    rem = _expected_remaining_outage(now)
    est = {DO_NOTHING: rem + MAIN_TIME,
           RETRY: max(RETRY_DELAY, rem) + MAIN_TIME,
           SUBSTITUTE: (len(env.backup.queue) + env.backup.count + 1) * BACKUP_TIME}
    if env.action_masks()[ESCALATE]:
        est[ESCALATE] = ESC_TIME
    return min(est, key=lambda a: (KAPPA[a] + MISS_PENALTY * (est[a] > slack), KAPPA[a]))


def policy_rule(env, threshold=6.0):
    """Hand-written rule (SmartPM / P20 style): urgent -> escalate if possible,
    else substitute; otherwise retry."""
    slack, _ = env.current_case()
    if slack < threshold:
        return ESCALATE if env.action_masks()[ESCALATE] else SUBSTITUTE
    return RETRY


def policy_fixed(action):
    def pol(env):
        if action == ESCALATE and not env.action_masks()[ESCALATE]:
            return RETRY
        return action
    return pol


BASELINES = {
    'do_nothing': policy_fixed(DO_NOTHING),
    'always_retry': policy_fixed(RETRY),
    'always_substitute': policy_fixed(SUBSTITUTE),
    'always_escalate': policy_fixed(ESCALATE),
    'greedy': policy_greedy,
    'rule': policy_rule,
}


def run_episode(env, policy, seed):
    """policy: callable(env) -> action, or an SB3 model (deterministic predict)."""
    obs, _ = env.reset(seed=seed)
    ret, done = 0.0, False
    while not done:
        if callable(policy):
            a = policy(env)
        else:
            a = int(policy.predict(obs, action_masks=np.array(env.action_masks()), deterministic=True)[0])
        obs, r, done, _, _ = env.step(a)
        ret += r
    return ret, dict(env.last_episode_info)
