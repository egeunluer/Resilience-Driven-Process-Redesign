#!/usr/bin/env python3
"""
Do-nothing sweep of the demand spike (D4), before it gets a decision trigger.

Every spike episode is paired with the same seed at arrival factor 1 (same warm-up,
onset and random draws up to the onset), so the differences are the spike's own
effect. Reported per factor: Loss, lowest 24 h rolling Q, time to recover, open
cases, expiries, the daily mean of Q after the onset, and how many work items wait
longer than the planned trigger (1 working day) -- the decisions the trigger would
create.

  python3 sweep_demand_spike.py                          # factor from the config (3)
  python3 sweep_demand_spike.py --factors 2 3 --seeds 16
"""
import argparse
import copy
import json
import os
from collections import Counter
from multiprocessing import Pool

import numpy as np

from loan_model.episode import Episode, HORIZON_DAYS
from loan_model.simulator import StageModel

ART = '../../Module 1/artifacts/'
SCENARIO = 'D4_demand_spike'
DAY = 86400.0


def _job(args):
    model_path, cfg, cal, factor, seed, wait_wd = args
    cfg = copy.deepcopy(cfg)
    lv = cfg['scenarios'][SCENARIO]['levels']['2']
    lv['arrival_factor'] = factor
    sm = StageModel(json.load(open(model_path)), pool=cal['pool'])
    ep = Episode(sm, cfg, SCENARIO, '2', seed, cal, record=True)
    sim, tr = ep.sim, ep.tr
    q_hourly, open_hourly = [], []

    def sample():
        while sim.env.now < ep.end:
            tr.advance(sim.env.now)
            q_hourly.append(tr.q)
            open_hourly.append(tr.open)
            yield sim.env.timeout(3600)
    sim.env.process(sample())
    ep.run()
    r = ep.result()
    arrivals = sum(1 for c in sim.cases.values() if ep.onset <= c['arrival'] < ep.onset + ep.duration_s)
    # work items requested after the onset that waited > wait_wd working days for a worker
    clock, s0, H = sim.clock, sim.start_s, clock_day(sim)
    late = Counter()
    for cid, item, w, t_req, t0, t1 in sim.sessions:
        if t_req >= ep.onset and clock.to_working(s0 + t0) - clock.to_working(s0 + t_req) > wait_wd * H:
            late[item] += 1
    q = np.array(q_hourly)
    days = len(q) // 24
    return {'factor': factor, 'seed': seed, 'duration_h': r['duration_h'], 'loss_q_days': r['loss_q_days'],
            'min_rolling_q': r['min_rolling_q'], 'time_to_recover_d': r['time_to_recover_d'],
            'recovered': r['recovered'], 'expiries': r['delay_expiries'], 'spike_arrivals': arrivals,
            'open_peak': int(max(open_hourly)), 'open_at_onset': int(open_hourly[0]),
            'q_daily': q[:days * 24].reshape(days, 24).mean(axis=1).tolist(),
            'late_items': dict(late), 'late_total': sum(late.values())}


def clock_day(sim):
    return (sim.h_max - sim.h_min) * 3600


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default=ART + 'bpic2017_stage_model.json')
    ap.add_argument('--disruptions', default=ART + 'bpic2017_disruptions.json')
    ap.add_argument('--calibration', default=ART + 'bpic2017_pool3_resilience_calibration.json')
    ap.add_argument('--factors', type=float, nargs='+')
    ap.add_argument('--seeds', type=int, default=16)
    ap.add_argument('--wait-working-days', type=float, default=1.0)
    ap.add_argument('--out')
    args = ap.parse_args()
    cfg = json.load(open(args.disruptions))
    cal = json.load(open(args.calibration))
    factors = args.factors or [cfg['scenarios'][SCENARIO]['levels']['2']['arrival_factor']]
    jobs = [(args.model, cfg, cal, f, s, args.wait_working_days) for f in [1.0] + factors for s in range(args.seeds)]
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        R = pool.map(_job, jobs, chunksize=1)
    if args.out:
        json.dump(R, open(args.out, 'w'))
    by = {(r['factor'], r['seed']): r for r in R}
    seeds = range(args.seeds)
    print(f"do-nothing, {args.seeds} seeds, horizon {HORIZON_DAYS} d, length drawn 72-120 h; "
          f"factor 1 = the same episodes without the spike")
    print(f"{'factor':>6s} {'Loss (Q-d)':>10s} {'sd':>6s} {'vs f=1':>14s} {'min rollQ':>9s} {'recover d':>9s} "
          f"{'expiries':>8s} {'extra arr.':>10s} {'open peak':>9s} {'waits>1wd':>9s}")
    for f in [1.0] + factors:
        rs = [by[(f, s)] for s in seeds]
        loss = np.array([r['loss_q_days'] for r in rs])
        dif = loss - np.array([by[(1.0, s)]['loss_q_days'] for s in seeds])
        h = 1.96 * dif.std(ddof=1) / np.sqrt(len(dif)) if f != 1.0 else 0.0
        extra = np.mean([r['spike_arrivals'] - by[(1.0, s)]['spike_arrivals'] for s, r in zip(seeds, rs)])
        print(f"{f:6g} {loss.mean():10.4f} {loss.std(ddof=1):6.4f} {dif.mean():+8.4f}±{h:5.4f} "
              f"{np.mean([r['min_rolling_q'] for r in rs]):9.3f} {np.mean([r['time_to_recover_d'] for r in rs]):9.1f} "
              f"{np.mean([r['expiries'] for r in rs]):8.1f} {extra:10.0f} {np.mean([r['open_peak'] for r in rs]):9.0f} "
              f"{np.mean([r['late_total'] for r in rs]):9.0f}")
    print(f"\ndaily mean Q after the onset (Q_ref {cal['q_ref']:.3f}), mean over seeds:")
    print('day   ' + ' '.join(f'{f:>6g}' for f in [1.0] + factors))
    n = min(len(r['q_daily']) for r in R)
    for d in list(range(0, 14)) + list(range(14, n, 4)):
        print(f'{d:3d}   ' + ' '.join(f"{np.mean([by[(f, s)]['q_daily'][d] for s in seeds]):6.3f}" for f in [1.0] + factors))
    print('\nwork items waiting > %g working day(s) for a worker (mean per episode):' % args.wait_working_days)
    items = sorted({i for r in R for i in r['late_items']})
    for i in items:
        print(f'  {i:32s} ' + ' '.join(f"{np.mean([by[(f, s)]['late_items'].get(i, 0) for s in seeds]):7.1f}"
                                       for f in [1.0] + factors))


if __name__ == '__main__':
    main()
