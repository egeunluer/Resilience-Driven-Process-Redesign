#!/usr/bin/env python3
"""
Calibrate the resilience measure of the stage-model simulator.

Phase 4 (this part): undisrupted runs give the normal behaviour of Q(t) -- its mean,
day-to-day spread, and Q_ref, the normal-operation floor below which Loss counts
(5th percentile of daily-mean Q) -- plus the Loss and delay-caused expiries that
accumulate per day without any disruption.
Phase 8 adds the do-nothing Loss per disruption scenario (Loss_base[e]).

  python3 calibrate_loan_resilience.py            # BPIC2017, seeds 0-15
"""
import argparse
import json
import os

import numpy as np

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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='../../Module 1/artifacts/bpic2017_stage_model.json')
    ap.add_argument('--out', default=None, help='default: <log>_resilience_calibration.json next to the model')
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(16)))
    ap.add_argument('--days', type=int, default=120, help='measured days per seed, after the warm-up')
    ap.add_argument('--pool-k', type=int, help='constrained pool: k most active workers per work item')
    args = ap.parse_args()
    sm = StageModel(args.model)
    tag = ''
    if args.pool_k:
        sm = StageModel(sm.m, pool=top_k_pool(sm.m, args.pool_k))
        tag = f'_pool{args.pool_k}'
    out = args.out or os.path.join(os.path.dirname(args.model), f"{sm.m['meta']['log']}{tag}_resilience_calibration.json")
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
