"""
One disruption episode on the stage-model simulator (Phase 5; the core of the
Phase 7 environment and of the Phase 8 calibration).

warm-up (WARMUP_DAYS) -> disruption onset at a random moment of working time in the
following week -> run for a fixed HORIZON_DAYS after onset. The horizon is fixed rather
than ending at recovery: a delay often makes cases miss their deadlines only weeks
later (telephony outage: 95% of the extra off-track time arrives by day ~40), and
recovery is declared long before that. Time to recover is still reported.

Decisions are answered either by `policy(sim, decision) -> action` (run_episode, e.g.
always do_nothing), or, with policy=None, by a controller that takes them from
`episode.sim.pending` (loan_env.LoanEnv) and lets the episode run on with
`run_to_decision()`.

The onset is drawn in working time: an outage that starts and ends at night blocks
nothing, so a wall-clock onset would give episodes without a single decision.
"""
from collections import Counter

import numpy as np

from loan_model.disruptions import DisruptionLayer
from loan_model.metrics import QTracker
from loan_model.simulator import Simulator

WARMUP_DAYS = 40
HORIZON_DAYS = 50
CHECK_EVERY_S = 3600.0


def do_nothing(sim, decision):
    return 'do_nothing'


class Episode:
    """Warm-up done and disruption scheduled on construction; the clock stands at the
    onset. `cal` is the resilience calibration (q_mean, deadline_floor_s, deadlines_s).
    `duration_h` fixes the disruption length (None: drawn from the level)."""

    def __init__(self, sm, disruption_cfg, scenario, level, seed, cal, policy=None,
                 duration_h=None, warmup_days=WARMUP_DAYS, horizon_days=HORIZON_DAYS, record=False):
        self.tr = QTracker(sm.m['deadlines'], sm.m['end_stages'], recover_q=cal['q_mean'],
                           deadline_floor_s=cal['deadline_floor_s'])
        self.sim = sim = Simulator(sm, seed=seed, observers=[self.tr], record=record)
        sim.decision_policy = policy
        self.seed = seed
        rng = np.random.default_rng(None if seed is None else seed + 7919)
        clock, s0 = sim.clock, sim.start_s
        w0 = clock.to_working(s0 + warmup_days * 86400)
        w1 = clock.to_working(s0 + (warmup_days + 7) * 86400)
        self.onset = onset = clock.from_working(float(rng.uniform(w0, w1))) - s0
        self.d = DisruptionLayer(disruption_cfg, scenario, level, cal['deadlines_s'], rng,
                                 duration_h=duration_h)
        sim.attach_disruption(self.d)
        self.duration_s = self.d.start(sim.env, onset)
        self.end = onset + horizon_days * 86400
        sim.env.run(until=onset)
        self.tr.reset_loss(onset)
        self.recovered_at = None
        self.min_rolling_q = self.tr.rolling_q(onset)
        self._end_ev = sim.env.timeout(self.end - onset)
        sim.env.process(self._watch())

    def _watch(self):
        """Hourly: recovery check and the lowest 24 h rolling Q."""
        env = self.sim.env
        while env.now < self.end:
            yield env.timeout(CHECK_EVERY_S)
            t = env.now
            self.min_rolling_q = min(self.min_rolling_q, self.tr.rolling_q(t))
            if self.recovered_at is None and self.tr.recovered(t, self.d.end):
                self.recovered_at = t

    @property
    def now(self):
        return self.sim.env.now

    @property
    def done(self):
        return self.now >= self.end

    def run_to_decision(self):
        """Run until a decision is pending (True) or the horizon is reached (False)."""
        sim = self.sim
        while not sim.pending and not self.done:
            sim.decision_ready = sim.env.event()
            sim.env.run(until=sim.decision_ready | self._end_ev)
        sim.decision_ready = None
        if self.done:
            self.tr.advance(self.end)
            return False
        self.tr.advance(self.now)
        return True

    def run(self):
        """Run to the horizon; decisions must be answered by the policy."""
        self.sim.env.run(until=self._end_ev)
        self.tr.advance(self.end)
        return self

    def result(self):
        sim, tr = self.sim, self.tr
        types = Counter(x[2] for x in sim.decisions)
        actions = Counter(x[5] for x in sim.decisions)
        return {
            'scenario': self.d.key, 'seed': self.seed, 'onset_day': self.onset / 86400,
            'onset_weekday': sim.local(self.onset).weekday(),
            'duration_h': self.duration_s / 3600,
            'recovered': self.recovered_at is not None,
            'time_to_recover_d': ((self.recovered_at or self.end) - self.onset) / 86400,
            'loss': tr.loss, 'loss_q_days': tr.loss / 86400,
            'min_rolling_q': self.min_rolling_q,
            'delay_expiries': tr.expiries,
            'decisions': len(sim.decisions),
            'cases_hit': len({x[1] for x in sim.decisions}),
            'decisions_by_type': dict(types), 'actions': dict(actions),
            'action_cost': sim.action_cost,
        }


def run_episode(sm, disruption_cfg, scenario, level, seed, cal, policy=do_nothing,
                duration_h=None, warmup_days=WARMUP_DAYS, horizon_days=HORIZON_DAYS):
    """Returns a dict of episode metrics."""
    return Episode(sm, disruption_cfg, scenario, level, seed, cal, policy=policy, duration_h=duration_h,
                   warmup_days=warmup_days, horizon_days=horizon_days).run().result()
