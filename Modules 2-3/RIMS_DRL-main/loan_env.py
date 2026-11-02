"""
Phase 7: gymnasium environment for the recovery-action agent on the stage-model
simulator (MaskablePPO compatible: action_masks()).

One episode = one disruption (loan_model.episode.Episode): warm-up, onset, fixed
horizon. Every step answers one decision -- a case blocked by the disruption, routed
into the documents loop during a surge, or an SLA-risk alert -- with a recovery action;
the simulation then runs on until the next decision or the horizon.

Reward between two decisions:
    r = -( w * dLoss / Loss_base[cell]  +  kappa[a] * c_p[cell] ),  c_p = 1 / N[cell]
dLoss is the Loss accrued since the previous decision (the Loss after the last
decision, up to the horizon, goes into the last step). Loss_base and N are the
do-nothing Loss and decisions per episode of the same cell (scenario, level, outage
length), from calibrate_loan_resilience.py (Phase 8): a do-nothing episode sums to -w,
and acting on every case costs about kappa in total.

Scenario selection per reset: a fixed `cycle` of (scenario, level, duration_h) cells
(duration_h None: drawn from the level), stepped through in order like the shock
cycle of gym_env; `shock_cycle_index` is reset by EvalPolicyCallback so evaluation
episode j always gets the same cell.
"""
import json

import numpy as np
from gymnasium import Env, spaces

from loan_model.disruptions import scenario_key
from loan_model.episode import Episode, HORIZON_DAYS, WARMUP_DAYS
from loan_model.simulator import StageModel

DAY = 86400.0
EMPTY_RETRIES = 10      # resets with no decision in the whole episode are redrawn


class LoanEnv(Env):
    metadata = {'render_modes': []}

    def __init__(self, stage_model, disruptions, calibration, cycle, w=1.0, norm=None,
                 warmup_days=WARMUP_DAYS, horizon_days=HORIZON_DAYS):
        """stage_model / disruptions / calibration: paths or dicts. The calibration's
        "pool" (constrained worker pool) is applied to the stage model. `norm` maps a
        cell key to {'loss_base': seconds, 'n_decisions': N}; default: the calibration's
        loss_base_by_scenario."""
        super().__init__()
        load = lambda x: json.load(open(x)) if isinstance(x, str) else x
        self.cal = load(calibration)
        self.cfg = load(disruptions)
        base = load(stage_model)
        self.sm = StageModel(base, pool=self.cal.get('pool'))
        self.cycle = [tuple(c) for c in cycle]
        self.w = w
        self.norm = norm if norm is not None else self.cal.get('loss_base_by_scenario', {})
        missing = [self._key(c) for c in self.cycle if self._key(c) not in self.norm]
        if missing:
            raise ValueError(f'no Loss_base / N for {missing}: run calibrate_loan_resilience.py (Phase 8) '
                             'or pass norm=')
        self.warmup_days, self.horizon_days = warmup_days, horizon_days
        rc = self.cfg['recovery']
        self.actions = list(rc['actions'])
        self.kappa = rc['kappa']
        self.max_retries = rc['retry'].get('max_per_case') or 1
        self.causes = sorted({s['failure_type'] for s in self.cfg['scenarios'].values()})
        self.alert_type = self.cfg.get('sla_alert', {}).get('failure_type', 'sla_risk')
        self.stages = [s for s in self.sm.m['stages'] if s not in self.sm.end_stages]
        self.scenarios = list(self.cfg['scenarios'])
        self.deadline = self.cal['deadlines_s']
        self.esc_capacity = rc['escalate']['capacity']
        self.action_space = spaces.Discrete(len(self.actions))
        self.obs_names = self._obs_names()
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(len(self.obs_names),), dtype=np.float32)
        self.shock_cycle_index = 0
        self.ep = None
        self.last_episode_info = {}
        self.empty_episodes = 0

    @staticmethod
    def _key(cell):
        return scenario_key(*cell)

    # ------------------------------------------------------------------ episode
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        cell = self.cycle[self.shock_cycle_index % len(self.cycle)]
        self.shock_cycle_index += 1
        scenario, level, duration_h = cell
        base_seed = seed if seed is not None else int(self.np_random.integers(2**31 - 1))
        for k in range(EMPTY_RETRIES):
            ep = Episode(self.sm, self.cfg, scenario, level, base_seed + 1_000_003 * k, self.cal,
                         duration_h=duration_h, warmup_days=self.warmup_days, horizon_days=self.horizon_days)
            if ep.run_to_decision():
                break
            self.empty_episodes += 1
        else:
            raise RuntimeError(f'no decision in {EMPTY_RETRIES} episodes of {self._key(cell)}')
        self.ep, self.cell = ep, cell
        n = self.norm[self._key(cell)]
        self.loss_base = n['loss_base']
        self.c_p = 1.0 / n['n_decisions']
        self._loss_mark = 0.0
        self.cost = 0.0
        self.steps = 0
        return self._obs(), {}

    def _decision(self):
        return self.ep.sim.pending[0]

    def action_masks(self):
        allowed = set(self.ep.sim.allowed_actions(self._decision()))
        return [a in allowed for a in self.actions]

    def step(self, action):
        ep = self.ep
        d = ep.sim.pending.popleft()
        name = self.actions[int(action)]
        if name not in ep.sim.allowed_actions(d):
            raise ValueError(f'action {name!r} is masked for {d}')
        d.event.succeed(name)
        # let the case act on it now, so e.g. an escalation holds its slot before the
        # mask of a decision pending at the same instant is computed
        ep.sim.env.run(until=d.event)
        self.steps += 1
        self.cost += self.kappa[name] * self.c_p
        pending = ep.run_to_decision()
        d_loss = ep.tr.loss - self._loss_mark
        self._loss_mark = ep.tr.loss
        r = -(self.w * d_loss / self.loss_base + self.kappa[name] * self.c_p)
        info = {}
        if not pending:
            res = ep.result()
            loss_ratio = res['loss'] / self.loss_base
            self.last_episode_info = {
                'objective': self.w * loss_ratio + self.cost,
                'loss_ratio': loss_ratio, 'cost_ratio': self.cost,
                'loss_q_days': res['loss_q_days'], 'time_to_recover_d': res['time_to_recover_d'],
                'recovered': float(res['recovered']), 'min_rolling_q': res['min_rolling_q'],
                'delay_expiries': res['delay_expiries'], 'decisions': res['decisions'],
                'cases_hit': res['cases_hit'], 'duration_h': res['duration_h'],
                **{f'n_{a}': res['actions'].get(a, 0) for a in self.actions},
            }
            info['resilience'] = {'shock': res['scenario'], **self.last_episode_info}
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._obs()
        return obs, float(r), not pending, False, info

    @property
    def mean_cycle_time(self):
        """Episode objective (lower is better), under the name EvalPolicyCallback reads."""
        return self.last_episode_info.get('objective', float('nan'))

    # -------------------------------------------------------------- observation
    def _obs_names(self):
        return (['disruption_active', 'since_onset', 'since_end', 'working_hours', 'hour_of_day',
                 'esc_free', 'pending', 'blocked', 'blocked_slack_lt2d', 'blocked_slack_2_7d',
                 'q_over_ref', 'rolling_q_over_ref', 'open_cases']
                + [f'scenario_{s}' for s in self.scenarios]
                + [f'cause_{c}' for c in self.causes] + ['alert']
                + [f'stage_{s}' for s in self.stages]
                + ['slack', 'age', 'expiry_left', 'attempts', 'retries_left', 'visit', 'offers', 'fraud'])

    def _obs(self):
        ep, sim, tr = self.ep, self.ep.sim, self.ep.tr
        d = self._decision()
        now, dis = ep.now, ep.d
        c = sim.cases[d.cid]
        # slack of the cases currently blocked by the disruption (who competes for the slots)
        slacks = []
        for cid in tr.hits:
            cs = tr.case.get(cid)
            if cs is not None and cs[1] in self.deadline:
                slacks.append(cs[0] + self.deadline[cs[1]] - now)
        slacks = np.array(slacks)
        n_blk = max(1, len(slacks))
        local = sim.local(now)
        g = [
            float(dis.active),
            min(1.0, (now - ep.onset) / (7 * DAY)),
            min(1.0, max(0.0, now - dis.end) / (7 * DAY)) if not dis.active else 0.0,
            float(bool(sim.on_shift)),
            (local.hour + local.minute / 60) / 24,
            (self.esc_capacity - sim.esc.count - len(sim.esc.queue)) / self.esc_capacity,
            min(1.0, len(sim.pending) / 100),
            min(1.0, len(tr.hits) / 500),
            float((slacks < 2 * DAY).sum()) / n_blk if len(slacks) else 0.0,
            float(((slacks >= 2 * DAY) & (slacks < 7 * DAY)).sum()) / n_blk if len(slacks) else 0.0,
            min(1.0, tr.q / tr.q_ref) * 2 - 1,
            min(1.0, tr.rolling_q(now) / tr.q_ref) * 2 - 1,
            min(1.0, tr.open / 4000),
        ]
        g += [float(dis.name == s) for s in self.scenarios]
        g += [float(d.cause == x) for x in self.causes] + [float(d.failure_type == self.alert_type)]
        g += [float(d.stage == s) for s in self.stages]
        dl = self.deadline.get(d.stage)
        slack = (c['arrival'] + dl - now) / dl if dl else 1.0
        exp = c.get('expiry_at')
        g += [
            float(np.clip(slack, -1, 1)),
            min(1.0, (now - c['arrival']) / (60 * DAY)),
            min(1.0, (exp - now) / (31 * DAY)) if exp is not None else 1.0,
            min(1.0, d.attempts / 5),
            max(0, self.max_retries - c.get('retries', 0)) / self.max_retries,
            min(1.0, sum(1 for s, _ in c['stages'] if s == d.stage) / 3),
            min(1.0, c['offers'] / 3),
            float('fraud' in c['flags']),
        ]
        return np.clip(np.array(g, dtype=np.float32), -1.0, 1.0)
