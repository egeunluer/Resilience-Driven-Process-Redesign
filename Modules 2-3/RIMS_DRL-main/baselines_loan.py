#!/usr/bin/env python3
"""
Phase 8: baseline ladder and paired benchmark for the recovery-action agent.

Policies (doc section 14), all `policy(sim, decision) -> action`:
  do_nothing          wait for the service / the normal loop
  always_<action>     retry / reassign / substitute / escalate whenever allowed, else do_nothing
  greedy              myopic: per case, the cheapest action whose expected delay keeps the
                      case within its current stage deadline (else the fastest one);
                      ignores the scarce slots and the effect on other cases
  rule                hand-written, SmartPM / P20 style: retry once; if the outage goes on
                      and the deadline is near, escalate (or substitute when no slot is
                      free); missing documents: re-fetch, chase when urgent
(The Metzger-style binary trigger -- a learned escalate-or-not -- is trained in Phase 9.)

Benchmark: every policy on the same episodes (cell x seed, seeds 200+), the training
cycle cells plus the robustness cells (length drawn). Reported per episode as the
environment's objective  w * Loss / Loss_base[e] + sum(kappa) / N[e]  (= minus the
episode return; Loss_base and N of the scenario, pooled over its cycle cells), with paired differences and 95% intervals.

  python3 baselines_loan.py                 # 8 seeds per cell
  python3 baselines_loan.py --seeds 16 --out bench.json
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

from loan_model.disruptions import cycle_key, sample_hours, scenario_key
from loan_model.episode import run_episode
from loan_model.simulator import StageModel

ART = '../../Module 1/artifacts/'
H = 3600.0
DAY = 86400.0


class Policy:
    def __init__(self, cfg, cal, sm):
        self.cfg, self.cal, self.sm = cfg, cal, sm
        self.rc = cfg['recovery']
        self.deadline = cal['deadlines_s']

    def slack(self, sim, d):
        c = sim.cases[d.cid]
        dl = self.deadline.get(d.stage)
        return c['arrival'] + dl - sim.env.now if dl else np.inf

    def pick(self, sim, d, prefs):
        """First allowed action of prefs, else do_nothing."""
        acts = sim.allowed_actions(d)
        return next((a for a in prefs if a in acts), 'do_nothing')


class Fixed(Policy):
    def __init__(self, action, *a):
        super().__init__(*a)
        self.action = action

    def __call__(self, sim, d):
        return self.pick(sim, d, [self.action])


class Greedy(Policy):
    MISS = 1.0          # > any kappa: avoiding a deadline miss is worth any single action

    def __init__(self, *a):
        super().__init__(*a)
        # belief about outage length: the level's distribution (the agent's prior too)
        rng = np.random.default_rng(0)
        self.lengths = {}
        for name, spec in self.cfg['scenarios'].items():
            self.lengths[name] = np.array([sample_hours(spec['levels']['2']['duration_h'], rng)
                                           for _ in range(20000)]) * H
        sd = self.sm.m['validation_targets']['stage_visit_duration']
        self.loop_s = float(sd['incomplete'][50] + sd['validation'][50])

    def remaining(self, sim):
        d = sim.disruption
        if not d.active:
            return 0.0
        el = sim.env.now - d.onset
        L = self.lengths[d.name]
        L = L[L > el]
        return float(L.mean() - el) if len(L) else 0.0

    def __call__(self, sim, d):
        rc, acts = self.rc, sim.allowed_actions(d)
        mean = lambda spec: float(np.mean(spec['uniform']))
        esc = mean(rc['escalate']['handling_working_h']) * H / rc['escalate']['success_p']
        if d.cause == 'documents_missing':
            up = sim.disruption.params.get('upload_delay_h')
            full = self.loop_s + (mean(up) * H if up else 0.0)
            est = {'do_nothing': full, 'retry': (1 - rc['retry']['documents_success_p']) * full,
                   'reassign': self.loop_s, 'escalate': (1 - rc['escalate']['success_p']) * full + esc}
        else:
            rem = self.remaining(sim)
            sub = rc['substitute'].get(d.cause, {})
            est = {'do_nothing': rem, 'retry': max(rc['retry']['delay_working_h'] * H, rem), 'escalate': esc,
                   'substitute': (mean(sub['extra_delay_h']) * H if 'extra_delay_h' in sub else 0.0)
                   + sub.get('rework_p', 0.0) * (mean(sub['rework_delay_h']) * H if 'rework_delay_h' in sub else 0.0)}
        slack = self.slack(sim, d)
        k = rc['kappa']
        return min((a for a in acts if a in est),
                   key=lambda a: (k[a] + self.MISS * (est[a] > slack), est[a], k[a]))


class Rule(Policy):
    def __init__(self, *a, urgent_days=7.0):
        super().__init__(*a)
        self.urgent = urgent_days * DAY

    def __call__(self, sim, d):
        urgent = self.slack(sim, d) < self.urgent
        if d.cause == 'documents_missing':
            return self.pick(sim, d, ['escalate', 'reassign'] if urgent else ['retry'])
        if d.failure_type != d.cause:            # SLA-risk alert: deadline close
            return self.pick(sim, d, ['escalate', 'substitute'])
        if d.attempts == 1:
            return self.pick(sim, d, ['retry'])
        return self.pick(sim, d, ['escalate', 'substitute'] if urgent else ['do_nothing'])


def make_policy(name, cfg, cal, sm):
    if name == 'do_nothing':
        return Fixed('do_nothing', cfg, cal, sm)
    if name.startswith('always_'):
        return Fixed(name[len('always_'):], cfg, cal, sm)
    return {'greedy': Greedy, 'rule': Rule}[name](cfg, cal, sm)


POLICIES = ['do_nothing', 'always_retry', 'always_reassign', 'always_substitute', 'always_escalate', 'greedy', 'rule']


def norm_of(r, norm):
    sc, lv = r['scenario'].split(':')[:2]
    return norm[cycle_key(sc, lv)]


def objective(r, norm, w=1.0):
    n = norm_of(r, norm)
    return w * r['loss'] / n['loss_base'] + r['action_cost'] / n['n_decisions']


_G = {}


def _init(model_path, cfg_path, cal_path):
    cal = json.load(open(cal_path))
    _G.update(cfg=json.load(open(cfg_path)), cal=cal,
              sm=StageModel(json.load(open(model_path)), pool=cal['pool']))


def _job(args):
    pol, cell, seed = args
    g = _G
    r = run_episode(g['sm'], g['cfg'], cell[0], cell[1], seed, g['cal'],
                    policy=make_policy(pol, g['cfg'], g['cal'], g['sm']), duration_h=cell[2])
    r['policy'] = pol
    return r


def ci(x):
    x = np.asarray(x, dtype=float)
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float('nan')


def report(R, norm, policies, cells, title):
    keys = [scenario_key(*c) for c in cells]
    by = {(r['policy'], r['scenario'], r['seed']): r for r in R}
    seeds = sorted({r['seed'] for r in R})
    print(f'\n== {title}: {len(keys)} cells x {len(seeds)} seeds; objective = Loss/Loss_base + cost/N '
          '(do-nothing ~1 over the cycle, lower is better)')
    print(f"{'policy':18s} {'objective':>9s} {'vs do-nothing':>18s} {'Loss ratio':>10s} {'cost':>6s} {'decisions':>9s}")
    for p in policies:
        rs = [by[(p, k, s)] for k in keys for s in seeds]
        ob = [objective(r, norm) for r in rs]
        dif = [objective(by[(p, k, s)], norm) - objective(by[('do_nothing', k, s)], norm) for k in keys for s in seeds]
        m, h = ci(dif)
        lr = np.mean([r['loss'] / norm_of(r, norm)['loss_base'] for r in rs])
        cost = np.mean([r['action_cost'] / norm_of(r, norm)['n_decisions'] for r in rs])
        print(f"{p:18s} {np.mean(ob):9.3f} {m:+9.3f} ± {h:6.3f} {lr:10.3f} {cost:6.3f} {np.mean([r['decisions'] for r in rs]):9.0f}")
    print('per cell (mean objective):')
    print(f"{'cell':32s} " + ' '.join(f'{p[:10]:>10s}' for p in policies))
    for k in keys:
        print(f'{k:32s} ' + ' '.join(f"{np.mean([objective(by[(p, k, s)], norm) for s in seeds]):10.3f}" for p in policies))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default=ART + 'bpic2017_stage_model.json')
    ap.add_argument('--disruptions', default=ART + 'bpic2017_disruptions.json')
    ap.add_argument('--calibration', default=ART + 'bpic2017_pool3_resilience_calibration.json')
    ap.add_argument('--seeds', type=int, default=8, help='seeds per cell, from --seed0')
    ap.add_argument('--seed0', type=int, default=200)
    ap.add_argument('--policies', nargs='+', default=POLICIES)
    ap.add_argument('--out')
    args = ap.parse_args()
    cfg = json.load(open(args.disruptions))
    norm = json.load(open(args.calibration))['loss_base_by_scenario']
    cycle = [tuple(c) for c in cfg['training_cycle']['cells']]
    robust = [tuple(c) for c in cfg.get('robustness_cells', {}).get('cells', [])]
    seeds = range(args.seed0, args.seed0 + args.seeds)
    jobs = [(p, c, s) for c in cycle + robust for s in seeds for p in args.policies]
    with Pool(max(1, (os.cpu_count() or 2) - 1), initializer=_init,
              initargs=(args.model, args.disruptions, args.calibration)) as pool:
        R = pool.map(_job, jobs, chunksize=1)
    if args.out:
        json.dump(R, open(args.out, 'w'))
    report(R, norm, args.policies, cycle, 'training cycle (fixed lengths)')
    if robust:
        report(R, norm, args.policies, robust, 'robustness (length drawn from the level)')


if __name__ == '__main__':
    main()
