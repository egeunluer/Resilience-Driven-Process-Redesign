#!/usr/bin/env python3
"""
BPIC 2019 preprocessing for the stage-model miner (Phase 2). The log has completion
events only, so the start of every staff work step is estimated (loan_model/start_times.py,
start-time repair with calendar and simultaneous batches) and written as a start event;
the result is an event table the miner reads like a start/complete log.

Scope (decisions of 2026-11-19, New_Approach_Implementation_Plan.md):
- item category "3-way match, invoice before GR" only;
- cases = purchase-order items; items with any event outside 2018-01-01 .. 2019-01-31
  are dropped (bad dates);
- the stage model is mined from items created Jan-Aug 2018 (later items are cut off by
  the end of the log); the open-case target is measured on all items of the category;
- batch users (and events without a user) are automated: their steps get no estimate
  and need no worker.

  python3 -m loan_model.prepare_bpic2019
"""
import json
import os

import numpy as np
import pandas as pd
from lxml import etree

from loan_model.start_times import estimate_starts
from mine_arrival_profile import weekly_rates

LOG = '../../Event Logs/BPI_Challenge_2019.xes'
MAP = '../../Module 1/artifacts/bpic2019_activity_map.json'
OUT = 'tmp_loan_training/bpic2019_events.parquet'
CATEGORY = '3-way match, invoice before GR'
WINDOW = ('2018-01-01', '2019-02-01')
MINE_CREATED_BEFORE = '2018-09-01'


def load():
    ev, cat = [], {}
    for _, tr in etree.iterparse(LOG, tag='{*}trace'):
        attrs, events = {}, []
        for c in tr:
            if c.tag.split('}')[-1] == 'event':
                a = {d.get('key'): d.get('value') for d in c}
                events.append((a.get('concept:name'), a.get('org:resource'), a.get('time:timestamp')))
            else:
                attrs[c.get('key')] = c.get('value')
        cid = attrs['concept:name']
        cat[cid] = attrs.get('Item Category')
        ev += [(cid, *e) for e in events]
        tr.clear()
        while tr.getprevious() is not None:
            del tr.getparent()[0]
    ev = pd.DataFrame(ev, columns=['case', 'activity', 'resource', 'time'])
    ev['time'] = pd.to_datetime(ev['time'], utc=True, format='ISO8601')
    return ev, cat


def q101(x):
    return [round(float(v), 1) for v in np.quantile(np.asarray(x, dtype=float), np.linspace(0, 1, 101))]


def payment_terms(ev, m):
    """Payment due dates (option 1, 2026-11-19). Due date = the vendor's invoice date
    ('Vendor creates invoice', a date) + the payment term. The log has no due date, only
    the payment run that paid the item: an item ready (invoice recorded, block removed)
    before run R_(k-1) but paid only in run R_k became due in (R_(k-1), R_k], so its term
    is drawn uniformly there, minus its invoice date (interval-censored). Items that
    became ready only after R_(k-1) may have been held up by their own processing and
    are not used. Returns the invoice-date offset from the PO-item creation and the
    term, as quantiles (s)."""
    rel = m['batch_release']
    tz = m['timezone']
    paid = ev[ev['activity'] == rel['activity']]
    day = paid['time'].dt.tz_convert(tz).dt.date
    cnt = day.value_counts()
    run_days = sorted(cnt[cnt >= rel['min_items_per_day']].index)
    g = paid.groupby(day)['time']
    run_end = {d: g.get_group(d).max() for d in run_days}
    first = lambda a: ev[ev['activity'] == a].groupby('case')['time'].min()
    last = lambda a: ev[ev['activity'] == a].groupby('case')['time'].max()
    df = pd.DataFrame({'po': first('Create Purchase Order Item'), 'inv': first('Vendor creates invoice'),
                       'clear': first(rel['activity'])})
    ready = pd.concat([last('Record Invoice Receipt'), last('Remove Payment Block')], axis=1).max(axis=1)
    df['ready'] = ready.reindex(df.index)
    df = df.dropna(subset=['po'])
    has_inv = df['inv'].notna()
    df['run'] = df['clear'].dt.tz_convert(tz).dt.date
    idx = {d: i for i, d in enumerate(run_days)}
    rng = np.random.default_rng(0)
    terms, bound = [], 0
    for r in df[has_inv & df['clear'].notna() & df['ready'].notna()].itertuples():
        k = idx.get(r.run)
        if not k:
            continue
        prev = run_end[run_days[k - 1]]
        if r.ready > prev:
            bound += 1
            continue
        due = prev + (r.clear - prev) * rng.random()
        terms.append((due - r.inv).total_seconds())
    off = (df.loc[has_inv, 'inv'] - df.loc[has_inv, 'po']).dt.total_seconds()
    out = {'_doc': payment_terms.__doc__.split('Returns')[0].strip(),
           'share_with_invoice_date': round(float(has_inv.mean()), 4),
           'invoice_offset_s': q101(off), 'term_s': q101(terms),
           'n_terms': len(terms), 'n_ready_bound': bound,
           'grace_s': 7 * 86400,
           '_grace': 'paid on time = by the first weekly payment run after the due date: late once unpaid 7 d after it'}
    print(f"payment terms from {len(terms)} items ({bound} ready only after the previous run, not used); "
          f"term days p10/p50/p90 {np.percentile(terms, [10, 50, 90]).round(-3) / 86400}; "
          f"invoice date offset from PO creation days p10/p50/p90 {np.percentile(off, [10, 50, 90]) / 86400}; "
          f"share with invoice date {has_inv.mean():.3f}")
    return out


def main():
    m = json.load(open(MAP))
    ev, cat = load()
    ev = ev[ev['case'].map(cat) == CATEGORY]
    lo, hi = (pd.Timestamp(x, tz='UTC') for x in WINDOW)
    bad = ev.loc[(ev['time'] < lo) | (ev['time'] >= hi), 'case'].unique()
    ev = ev[~ev['case'].isin(bad)].sort_values(['case', 'time'], kind='stable').reset_index(drop=True)
    print(f'{ev.case.nunique()} items of "{CATEGORY}" ({len(bad)} dropped for dates outside {WINDOW})')

    # open items over the year, all items of the category (validation target)
    g = ev.groupby('case')['time']
    st, en = g.min(), g.max()
    grid = pd.date_range('2018-03-01', '2018-09-01', freq='D', tz='UTC') + pd.Timedelta(hours=12)
    wip = pd.Series([int(((st <= t) & (en > t)).sum()) for t in grid])

    # arrivals: creation of the purchase-order item, all items of the category, over 2018
    po = ev[ev['activity'] == 'Create Purchase Order Item'].groupby('case')['time'].min()
    po = po[(po >= lo) & (po < pd.Timestamp('2019-01-01', tz='UTC'))]
    loc = sorted(po.dt.tz_convert(m['timezone']).dt.tz_localize(None).dt.to_pydatetime())
    span_d = (loc[-1] - loc[0]).total_seconds() / 86400
    arrivals = {'rates_per_hour_of_week': [round(r, 6) for r in weekly_rates(loc)], 'per_day': round(len(loc) / span_d, 3),
                '_doc': 'creation of the purchase-order item, all items of the category created in 2018'}

    ev = ev[ev['case'].isin(st[st < pd.Timestamp(MINE_CREATED_BEFORE, tz='UTC')].index)].reset_index(drop=True)
    terms = payment_terms(ev, m)
    items = set(m['stage_work_items'].values())
    res = ev['resource'].fillna('NONE')
    automated = res.str.startswith('batch') | (res == 'NONE')
    ev['staff'] = ev['activity'].isin(items) & ~automated
    cal = {'timezone': m['timezone'], 'hour_min': m['estimation_calendar']['hour_min']}
    est = estimate_starts(ev, 'calendar_batch', cal)
    ev['start'] = est.reset_index(drop=True)
    # every work-item event gets a start (automated ones: start = completion), so that
    # "<item>@start" milestones mark the stage entry for automated steps too
    work = ev['activity'].isin(items)
    ev.loc[work & ev['start'].isna(), 'start'] = ev.loc[work & ev['start'].isna(), 'time']
    ev['resource'] = res.where(~automated, 'automated')
    starts = ev[work][['case', 'activity', 'resource', 'start']].rename(columns={'start': 'time'})
    starts['time'] = pd.to_datetime(starts['time'], utc=True)
    starts['lifecycle'] = 'start'
    ends = ev[['case', 'activity', 'resource', 'time']].copy()
    ends['lifecycle'] = 'complete'
    out = pd.concat([starts, ends]).sort_values(['case', 'time', 'lifecycle'], ascending=[True, True, False],
                                                kind='stable')
    out = out[['case', 'activity', 'lifecycle', 'resource', 'time']].reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_parquet(OUT)
    d = (ev.loc[ev['staff'], 'time'] - ev.loc[ev['staff'], 'start']).dt.total_seconds() / 60
    print(f'{ev.case.nunique()} items created before {MINE_CREATED_BEFORE} -> {OUT} ({len(out)} events); '
          f'estimated staff steps: {int(ev.staff.sum())}, minutes p50 by step:')
    print(d.groupby(ev.loc[ev['staff'], 'activity']).describe(percentiles=[.5, .9])[['count', '50%', '90%']].round(1))
    print(f'open items (all items, Mar-Aug 2018, daily at noon): median {int(wip.median())}, '
          f'p10 {int(wip.quantile(.1))}, p90 {int(wip.quantile(.9))}')
    print(f"arrivals: {arrivals['per_day']}/day")
    json.dump({'arrivals': arrivals, 'payment_terms': terms,
               'validation_targets': {'arrivals_per_day': arrivals['per_day'],
                                      'open_cases': {'median': int(wip.median()), 'p10': int(wip.quantile(.1)),
                                                     'p90': int(wip.quantile(.9))}}},
              open(OUT.replace('.parquet', '_targets.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
