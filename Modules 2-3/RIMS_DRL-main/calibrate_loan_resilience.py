#!/usr/bin/env python3
"""
Calibrate the resilience measure of the stage-model simulator.

Phase 4 (this part): undisrupted runs give the normal behaviour of Q(t) -- its mean,
day-to-day spread, and Q_ref, the normal-operation floor below which Loss counts
(5th percentile of daily-mean Q) -- plus the Loss and delay-caused expiries that
accumulate per day without any disruption.
Phase 8 (--scenarios): the do-nothing Loss and number of decisions per episode for
every cell (scenario, level, outage length) of the disruption config's training cycle
and robustness cells, and their mean over each scenario's cycle cells -- Loss_base and
N of the environment's reward.

  python3 calibrate_loan_resilience.py --pool-k 3               # Q_ref etc., seeds 0-15
  python3 calibrate_loan_resilience.py --pool-k 3 --scenarios   # Loss_base / N per cell
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

from loan_model.disruptions import cycle_key, scenario_key
from loan_model.episode import run_episode
from loan_model.metrics import QTracker
from loan_model.simulator import Simulator, StageModel, top_k_pool

WARMUP_DAYS = 45
DEADLINE_FLOOR_S = 86400.0


def daily_q(sm, seed, days):
    tr = QTracker(sm.m['deadlines'], sm.m['end_stages'], deadline_floor_s=DEADLINE_FLOOR_S)
    sim = Simulator(sm, seed=seed, observers=[tr], record=False).run(WARMUP_DAYS)
    out, expiries = [], 0
    for d in range(WARMUP_DAYS, WARMUP_DAYS + days):
        tr.reset_loss(d * 86400)
        sim.run(d + 1)
        tr.advance((d + 1) * 86400)
        out.append(tr.q_area / 86400)
        expiries += tr.expiries
    return out, expiries


def _do_nothing_job(args):
    model, cfg, cal, (scenario, level, duration_h), seed = args
    sm = StageModel(model, pool=cal['pool'])
    return run_episode(sm, cfg, scenario, level, seed, cal, duration_h=duration_h)


def calibrate_scenarios(model_path, disruptions_path, cal_path, seeds):
    """Do-nothing Loss_base and N (decisions per episode) per cell, written into the
    calibration file's loss_base_by_scenario."""
    with open(model_path) as f:
        model = json.load(f)
    with open(disruptions_path) as f:
        cfg = json.load(f)
    with open(cal_path) as f:
        cal = json.load(f)
    cells = [tuple(c) for c in cfg['training_cycle']['cells'] + cfg.get('robustness_cells', {}).get('cells', [])]
    jobs = [(model, cfg, cal, c, s) for c in cells for s in seeds]
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        res = pool.map(_do_nothing_job, jobs, chunksize=1)
    out = {}
    print(f"{'cell':32s} {'Loss_base (Q-days)':>18s} {'sd':>7s} {'N decisions':>11s} {'recover (d)':>11s}")
    for c in cells:
        key = scenario_key(*c)
        rs = [r for r in res if r['scenario'] == key]
        loss = np.array([r['loss'] for r in rs])
        n = np.array([r['decisions'] for r in rs])
        out[key] = {'loss_base': float(loss.mean()), 'loss_sd': float(loss.std(ddof=1)),
                    'n_decisions': float(n.mean()), 'n_seeds': len(rs),
                    'time_to_recover_d': float(np.mean([r['time_to_recover_d'] for r in rs]))}
        print(f"{key:32s} {loss.mean() / 86400:18.4f} {loss.std(ddof=1) / 86400:7.4f} {n.mean():11.0f} "
              f"{out[key]['time_to_recover_d']:11.1f}")
    # per scenario: pooled over its cycle cells. The environment normalises by these:
    # short outages do almost no harm (Loss at the level of the normal day-to-day
    # noise), so dividing by a cell's own Loss_base would blow that noise up.
    for (sc, lv) in dict.fromkeys((c[0], c[1]) for c in cfg['training_cycle']['cells']):
        ks = [scenario_key(*c) for c in cfg['training_cycle']['cells'] if (c[0], c[1]) == (sc, lv)]
        key = cycle_key(sc, lv)
        out[key] = {'loss_base': float(np.mean([out[k]['loss_base'] for k in ks])),
                    'n_decisions': float(np.mean([out[k]['n_decisions'] for k in ks])), 'cells': ks}
        print(f"{key:32s} {out[key]['loss_base'] / 86400:18.4f} {'':7s} {out[key]['n_decisions']:11.0f}   (cycle mean)")
    cal['loss_base_by_scenario'] = out
    cal['loss_base_definition'] = ('mean do-nothing Loss (seconds, Q_ref-normalised) and decisions per episode '
                                   f'per cell, seeds {seeds[0]}-{seeds[-1]}, warm-up/horizon of loan_model.episode')
    with open(cal_path, 'w') as f:
        json.dump(cal, f, indent=1)
    print(f'-> {cal_path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='../../Module 1/artifacts/bpic2017_stage_model.json')
    ap.add_argument('--out', default=None, help='default: <log>_resilience_calibration.json next to the model')
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(16)))
    ap.add_argument('--days', type=int, default=120, help='measured days per seed, after the warm-up')
    ap.add_argument('--pool-k', type=int, help='constrained pool: k most active workers per work item')
    ap.add_argument('--scenarios', action='store_true', help='Phase 8: do-nothing Loss_base / N per cell')
    ap.add_argument('--disruptions', default='../../Module 1/artifacts/bpic2017_disruptions.json')
    args = ap.parse_args()
    sm = StageModel(args.model)
    tag = ''
    if args.pool_k:
        sm = StageModel(sm.m, pool=top_k_pool(sm.m, args.pool_k))
        tag = f'_pool{args.pool_k}'
    out = args.out or os.path.join(os.path.dirname(args.model), f"{sm.m['meta']['log']}{tag}_resilience_calibration.json")
    if args.scenarios:
        calibrate_scenarios(args.model, args.disruptions, out, args.seeds)
        return
    daily, exp = [], 0
    for s in args.seeds:
        q, e = daily_q(sm, s, args.days)
        daily += q
        exp += e
    daily = np.array(daily)
    q_ref = float(np.percentile(daily, 5))
    n_days = len(daily)
    # Loss per day that undisrupted operation accumulates below Q_ref (daily means
    # under-state it slightly, since Q also moves within a day)
    base_loss = float(np.mean(np.maximum(0.0, q_ref - daily) / q_ref)) * 86400
    cal = {
        'model': os.path.basename(args.model),
        'pool_k': args.pool_k, 'pool': sm.pool,
        'seeds': args.seeds, 'warmup_days': WARMUP_DAYS, 'measured_days_per_seed': args.days,
        'deadline_floor_s': DEADLINE_FLOOR_S,
        'deadlines_s': {k: max(v, DEADLINE_FLOOR_S) for k, v in sm.m['deadlines'].items()},
        'q_mean': float(daily.mean()), 'q_daily_sd': float(daily.std()),
        'q_daily_p1': float(np.percentile(daily, 1)), 'q_ref': q_ref,
        'q_ref_definition': '5th percentile of daily-mean Q(t) in undisrupted runs',
        'baseline_loss_s_per_day': base_loss,
        'baseline_delay_expiries_per_day': exp / n_days,
        'loss_base_by_scenario': {},
    }
    if os.path.exists(out):
        with open(out) as f:
            cal['loss_base_by_scenario'] = json.load(f).get('loss_base_by_scenario', {})
    with open(out, 'w') as f:
        json.dump(cal, f, indent=1)
    print(f"{n_days} days ({len(args.seeds)} seeds): Q mean {cal['q_mean']:.4f}, daily sd {cal['q_daily_sd']:.4f}, "
          f"Q_ref (p5) {q_ref:.4f}, p1 {cal['q_daily_p1']:.4f}; baseline Loss {base_loss:.0f} s/day, "
          f"delay expiries {cal['baseline_delay_expiries_per_day']:.2f}/day -> {out}")


if __name__ == '__main__':
    main()
