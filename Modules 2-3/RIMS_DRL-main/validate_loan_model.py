#!/usr/bin/env python3
"""
Phase 3: validate the base (undisrupted) stage-model simulator against the log it
was mined from (doc §6: the simulator must reproduce the log before any disruption
is injected). Compares cycle times, outcomes, loops, offers, open cases, stage visit
durations and arrivals, and reports worker utilisation and queueing for workers.

  python3 validate_loan_model.py                       # BPIC2017, 3 seeds x 240 days
  python3 validate_loan_model.py --model "../../Module 1/artifacts/bpi2012_stage_model.json"

Case statistics use cases that arrive between day WARMUP and day WARMUP + WINDOW,
so all of them have had time to finish before the end of the run.
"""
import argparse
import json
import time
from collections import Counter

import numpy as np

from loan_model.simulator import Simulator, StageModel, top_k_pool

WARMUP, WINDOW = 45, 105


def run_one(sm, seed, days):
    t = time.time()
    sim = Simulator(sm, seed=seed).run(days)
    return sim, time.time() - t


def collect(sims, days):
    lo, hi = WARMUP * 86400, (WARMUP + WINDOW) * 86400
    cyc, outc, inc, rounds, offers, timers, unfinished = [], Counter(), 0, Counter(), Counter(), 0, 0
    stage_dur = {}
    n = 0
    open_cases, arrivals_per_day = [], []
    qwait, work, shift = {}, 0.0, 0.0
    for sim in sims:
        for c in sim.cases.values():
            if not lo <= c['arrival'] < hi:
                continue
            if c['end'] is None:
                unfinished += 1
                continue
            n += 1
            cyc.append(c['end'] - c['arrival'])
            outc[c['outcome']] += 1
            st = [s for s, _ in c['stages']]
            inc += 'incomplete' in st
            rounds[str(st.count('validation'))] += 1
            offers[str(c['offers'])] += 1 if c['offers'] else 0
            timers += bool(c.get('timers'))
            for (s, t0), (_, t1) in zip(c['stages'], c['stages'][1:]):
                stage_dur.setdefault(s, []).append(t1 - t0)
        grid = np.arange(60 * 86400 + 43200, days * 86400, 86400)
        arr = np.array([c['arrival'] for c in sim.cases.values()])
        end = np.array([c['end'] if c['end'] is not None else np.inf for c in sim.cases.values()])
        open_cases += [int(((arr <= g) & (end > g)).sum()) for g in grid]
        arrivals_per_day.append(((arr >= lo) & (arr < hi)).sum() / WINDOW)
        for cid, item, w, t_req, t0, t1 in sim.sessions:
            if t_req >= lo:
                qwait.setdefault(item, []).append(t0 - t_req)
                work += t1 - t0
        shift += sum(sim.shift_seconds.values()) * (days - WARMUP) / days
    return dict(n=n, unfinished=unfinished, cyc=np.array(cyc), outc=outc, inc=inc, rounds=rounds, offers=offers,
                timers=timers, stage_dur=stage_dur, open_cases=np.array(open_cases),
                arrivals=float(np.mean(arrivals_per_day)), qwait=qwait, util=work / shift if shift else float('nan'))


def ks_vs_quantiles(sample, log_q):
    """KS distance between the simulated sample's ECDF and the log's quantile function."""
    sample = np.sort(sample)
    k = np.linspace(0, 1, len(log_q))
    f_sim = np.searchsorted(sample, log_q, side='right') / len(sample)
    return float(np.max(np.abs(f_sim - k)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='../../Module 1/artifacts/bpic2017_stage_model.json')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--days', type=int, default=240)
    ap.add_argument('--out', help='write the comparison as JSON')
    ap.add_argument('--pool-k', type=int, help='constrained pool: k most active workers per work item')
    args = ap.parse_args()
    sm = StageModel(args.model)
    if args.pool_k:
        sm = StageModel(sm.m, pool=top_k_pool(sm.m, args.pool_k))
        print(f'constrained pool: {len(sm.workers)} workers (top {args.pool_k} per work item)')
    vt = sm.m['validation_targets']
    sims, walls = [], []
    for s in args.seeds:
        sim, w = run_one(sm, s, args.days)
        sims.append(sim)
        walls.append(w)
    r = collect(sims, args.days)
    D = 86400
    pct = lambda x, p: float(np.percentile(x, p))
    rows = []

    def row(name, sim_v, log_v, fmt='{:.2f}', ok=None):
        rows.append((name, fmt.format(sim_v), fmt.format(log_v), '' if ok is None else ('ok' if ok else 'CHECK')))

    print(f"{sm.m['meta']['log']}: {len(args.seeds)} seeds x {args.days} days, wall {np.mean(walls):.1f} s per run; "
          f"{r['n']} finished cases arriving on days {WARMUP}-{WARMUP + WINDOW} ({r['unfinished']} unfinished)\n")
    for p in (10, 25, 50, 75, 90, 95):
        sv, lv = pct(r['cyc'], p) / D, vt['cycle_time'][f'p{p}'] / D
        row(f'cycle time p{p} (days)', sv, lv, ok=abs(sv - lv) <= 0.1 * lv if p in (50, 90) else None)
    ks = ks_vs_quantiles(r['cyc'], vt['cycle_time_quantiles'])
    row('cycle time KS distance', ks, 0.0, '{:.3f}', ok=ks <= 0.1)
    for o in ('END_accepted', 'END_cancelled', 'END_denied'):
        sv, lv = r['outc'][o] / r['n'], vt['outcome'].get(o, 0)
        row(f'outcome {o} share', sv, lv, '{:.3f}', ok=abs(sv - lv) <= 0.03)
    row('cases with incomplete loop', r['inc'] / r['n'], vt['incomplete_case_share'], '{:.3f}',
        ok=abs(r['inc'] / r['n'] - vt['incomplete_case_share']) <= 0.03)
    for k in ('0', '1', '2', '3'):
        row(f'validation rounds = {k}', r['rounds'][k] / r['n'], vt['validation_rounds'].get(k, 0), '{:.3f}')
    with_off = sum(r['offers'].values())
    for k in ('1', '2', '3'):
        row(f'offers per case = {k} (cases with offers)', r['offers'][k] / with_off if with_off else 0,
            vt['offers_per_case'].get(k, 0) / max(1e-9, 1 - vt['offers_per_case'].get('0', 0)), '{:.3f}')
    row('open cases median', float(np.median(r['open_cases'])), vt['open_cases']['median'], '{:.0f}',
        ok=abs(np.median(r['open_cases']) - vt['open_cases']['median']) <= 0.1 * vt['open_cases']['median'])
    row('open cases p10', pct(r['open_cases'], 10), vt['open_cases']['p10'], '{:.0f}')
    row('open cases p90', pct(r['open_cases'], 90), vt['open_cases']['p90'], '{:.0f}')
    row('arrivals per day', r['arrivals'], vt['arrivals_per_day'], '{:.1f}', ok=abs(r['arrivals'] - vt['arrivals_per_day']) <= 0.05 * vt['arrivals_per_day'])
    for s, q in vt['stage_visit_duration'].items():
        if q and s in r['stage_dur']:
            row(f'{s} visit p50 (days)', pct(r['stage_dur'][s], 50) / D, q[50] / D, '{:.2f}')
            row(f'{s} visit p90 (days)', pct(r['stage_dur'][s], 90) / D, q[90] / D, '{:.2f}')
    w = max(len(x[0]) for x in rows)
    print(f"{'metric':{w}s}  {'simulated':>10s}  {'log':>10s}  check")
    for name, sv, lv, ok in rows:
        print(f'{name:{w}s}  {sv:>10s}  {lv:>10s}  {ok}')
    print(f"\nworker utilisation (work / shift time): {r['util']:.3f}")
    print('queueing for a worker (request -> session start), hours:')
    for item, q in sorted(r['qwait'].items()):
        q = np.array(q) / 3600
        print(f'  {item:28s} p50 {np.percentile(q, 50):6.2f}  p90 {np.percentile(q, 90):6.2f}  share waiting > 1h {np.mean(q > 1):.2f}')
    n_ok = sum(1 for x in rows if x[3] == 'ok'); n_chk = sum(1 for x in rows if x[3] == 'CHECK')
    print(f'\n{n_ok} checks ok, {n_chk} to check')
    if args.out:
        json.dump({'rows': rows, 'utilisation': r['util']}, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
