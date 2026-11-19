"""
Start-time estimation for completion-only event logs (BPIC 2019), after the start-time
repair method of Chapela-Campa, López-Pintado, Dumas et al. (2022, "Repairing Activity
Start Times to Improve Business Process Simulation"):

  start = max(enablement, availability), clipped to the completion

- enablement (Wombacher & Iacob): completion of the case's closest preceding event whose
  activity is not concurrent with this one. Concurrency oracle: two activities are
  concurrent if each directly follows the other somewhere in the log.
- availability (Zabka et al.): the worker's previous completion; with a calendar, no
  earlier than the start of the worker's working day (off-duty time is not work).
- batches (Martin et al. 2015, simultaneous batch): events of one worker for one
  activity with the same completion time were handled together. The batch gets one
  start (latest enablement of its items, the worker's availability before the batch)
  and its duration is split evenly over its items.

Variants for the check in validate_start_times.py: naive (since the case's previous
event), oracle (Module 1's concurrency_oracle_estimator: no calendar, no batches),
calendar, calendar + batches.
"""
from collections import defaultdict

import numpy as np
import pandas as pd


def concurrency_oracle(seqs):
    """{frozenset(a, b)} of activities that directly follow each other in both orders."""
    fwd = set()
    for acts in seqs:
        for a, b in zip(acts, acts[1:]):
            if a != b:
                fwd.add((a, b))
    return {frozenset(p) for p in fwd if (p[1], p[0]) in fwd}


def _enablement(ev, conc, naive=False):
    """Completion of the closest preceding non-concurrent event of the same case (seconds)."""
    out = np.full(len(ev), np.nan)
    acts, ts = ev['activity'].to_numpy(), ev['t'].to_numpy()
    for idx in ev.groupby('case', sort=False).indices.values():
        idx = np.sort(idx)
        for k, i in enumerate(idx):
            for j in idx[:k][::-1]:
                if ts[j] > ts[i]:
                    continue
                if not naive and acts[j] != acts[i] and frozenset((acts[j], acts[i])) in conc:
                    continue
                out[i] = ts[j]
                break
    return out


def estimate_starts(ev, variant='calendar_batch', calendar=None):
    """ev: DataFrame with case, activity, resource, time (UTC completion) and a boolean
    `staff` column (events whose start is estimated; all events give enablement).
    Rows must be sorted by case, then time. Returns estimated start times (UTC) for the
    staff rows, NaT elsewhere."""
    ev = ev.reset_index(drop=True).copy()
    t0 = ev['time'].min()
    ev['t'] = (ev['time'] - t0).dt.total_seconds()
    seqs = ev.groupby('case', sort=False)['activity'].agg(list)
    conc = set() if variant == 'naive' else concurrency_oracle(seqs)
    enable = _enablement(ev, conc, naive=(variant == 'naive'))
    start = np.full(len(ev), np.nan)
    staff = ev['staff'].to_numpy()
    if variant == 'naive':
        start[staff] = enable[staff]
    else:
        s = ev[staff].copy()
        s['enable'] = enable[staff]
        use_batch = variant == 'calendar_batch'
        key = ['resource', 'activity', 't'] if use_batch else ['resource', 't', 'row']
        s['row'] = s.index
        grp = s.groupby(key, sort=False).agg(enable=('enable', 'max'), rows=('row', list)).reset_index()
        grp = grp.sort_values(['resource', 't'], kind='stable')
        # the worker's previous completion before this event / batch
        prev = grp.groupby('resource')['t'].shift(1).to_numpy()
        avail = prev.copy()
        if variant in ('calendar', 'calendar_batch'):
            loc = (t0 + pd.to_timedelta(grp['t'], unit='s')).dt.tz_convert(calendar['timezone'])
            day_start = loc.dt.normalize() + pd.Timedelta(hours=calendar['hour_min'])
            ds = (day_start.dt.tz_convert('UTC') - t0).dt.total_seconds().to_numpy()
            avail = np.where(np.isnan(prev), ds, np.maximum(prev, ds))
        st = np.fmax(grp['enable'].to_numpy(), avail)          # max, ignoring a missing side
        st = np.minimum(st, grp['t'].to_numpy())
        dur = grp['t'].to_numpy() - st
        for rows, t, d in zip(grp['rows'], grp['t'], dur):
            per = d / len(rows) if use_batch else d
            for r in rows:
                start[r] = t - per
    return t0 + pd.to_timedelta(pd.Series(start), unit='s')
