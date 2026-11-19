#!/usr/bin/env python3
"""
Check the start-time estimation (loan_model/start_times.py) on logs whose real work
sessions are known (BPIC 2017, BPI 2012): hide every start/resume, keep only the
completion of each session (suspend / complete / ate_abort) and of every other event,
estimate the session starts and compare with the real sessions, as mined by
loan_model/mine_stage_model.py (the durations the simulator uses).

  python3 validate_start_times.py --log bpic2017
  python3 validate_start_times.py --log bpi2012
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from loan_model.mine_stage_model import load_log, sessions_of
from loan_model.start_times import estimate_starts

ART = '../../Module 1/artifacts/'
VARIANTS = ['naive', 'oracle', 'calendar', 'calendar_batch']


def hidden_log(m, ev):
    """Completion-only view of the log plus the real session of each staff event."""
    system = set(m['system_resources'])
    lc_open, lc_close = set(m['lifecycle']['open']), set(m['lifecycle']['close'])
    items = set(m['stage_work_items'].values()) | set(m.get('optional_work_items', {}))
    ignored = set(m.get('ignored_activities', []))
    ev = ev[~ev['activity'].isin(ignored)]
    rows = []
    for cid, g in ev.groupby('case', sort=False):
        acts, lcs, ress, ts = g['activity'].tolist(), g['lifecycle'].tolist(), g['resource'].tolist(), g['time'].tolist()
        true = {(a, r, t1): t0 for a, r, t0, t1 in sessions_of(zip(acts, lcs, ress, ts), lc_open, lc_close, system)}
        for a, lc, r, t in zip(acts, lcs, ress, ts):
            if lc in lc_open or lc in ('schedule', 'withdraw'):
                continue                                  # hidden: only completions remain
            t0 = true.get((a, r, t)) if a in items else None
            rows.append((cid, a, r, t, t0 is not None, t0))
    h = pd.DataFrame(rows, columns=['case', 'activity', 'resource', 'time', 'staff', 'true_start'])
    return h.sort_values(['case', 'time'], kind='stable').reset_index(drop=True)


def report(h, est, name):
    s = h[h['staff']].copy()
    s['est'] = (s['time'] - est[s.index]).dt.total_seconds()
    s['true'] = (s['time'] - s['true_start']).dt.total_seconds()
    s = s[s['true'] > 0]
    ratio = (s['est'] + 1) / (s['true'] + 1)
    pair = s.groupby(['resource', 'activity']).agg(est=('est', 'median'), true=('true', 'median'), n=('est', 'size'))
    pair = pair[pair['n'] >= 3]
    pr = (pair['est'] + 1) / (pair['true'] + 1)
    out = {'variant': name, 'events': len(s),
           'event_median_abs_err_min': float((s['est'] - s['true']).abs().median() / 60),
           'event_median_ratio': float(ratio.median()),
           'event_within_2x': float(((ratio >= 0.5) & (ratio <= 2)).mean()),
           'pairs': len(pair), 'pair_median_ratio': float(pr.median()),
           'pair_within_2x': float(((pr >= 0.5) & (pr <= 2)).mean()),
           'pair_log_corr': float(np.corrcoef(np.log1p(pair['est']), np.log1p(pair['true']))[0, 1])}
    per_item = s.groupby('activity').agg(est_p50=('est', 'median'), true_p50=('true', 'median'),
                                         est_p90=('est', lambda x: x.quantile(.9)),
                                         true_p90=('true', lambda x: x.quantile(.9)))
    return out, (per_item / 60).round(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--log', required=True, choices=['bpic2017', 'bpi2012'])
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    m = json.load(open(ART + f'{args.log}_activity_map.json'))
    cal = json.load(open(ART + f'{args.log}_stage_model.json'))['calendar']
    repo = os.path.abspath(os.path.join(ART, '..', '..'))
    ev, _ = load_log(os.path.join(repo, m['log_path']), [])
    h = hidden_log(m, ev)
    print(f"{args.log}: {len(h)} completion events, {int(h['staff'].sum())} work sessions to estimate; "
          f"calendar {cal['hour_min']}-{cal['hour_max']} h {cal['timezone']}")
    rows = []
    for v in VARIANTS:
        out, per_item = report(h, estimate_starts(h, v, cal), v)
        rows.append(out)
        print(f"\n[{v}] per work item, minutes (estimated vs real session):\n{per_item.to_string()}")
    df = pd.DataFrame(rows).set_index('variant')
    print('\n' + df.round(3).to_string())
    if args.out:
        df.to_json(args.out, indent=1)


if __name__ == '__main__':
    main()
