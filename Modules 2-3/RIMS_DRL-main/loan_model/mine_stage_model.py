#!/usr/bin/env python3
"""
Mine a stage-model config from an event log, driven by a log-specific activity map
(schema and semantics: loan_model/STAGE_MODEL_SCHEMA.md). Generic: nothing here is
specific to BPIC2017 -- another log needs only its own activity map.

Work is measured as active sessions (open lifecycle -> close lifecycle, e.g. start/
resume -> suspend/complete/ate_abort), not as start -> complete of one resource:
in BPIC2017 a work item is suspended for days while the bank waits for the customer,
so start -> complete (Module 1's true-PT pairing) mixes waiting into work time. The
suspended gaps become the resource-free waits of the stage model.

Usage (from Modules 2-3/RIMS_DRL-main):
  python3 -m loan_model.mine_stage_model \\
      --map "../../Module 1/artifacts/bpic2017_activity_map.json" \\
      --out "../../Module 1/artifacts/bpic2017_stage_model.json"
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from lxml import etree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mine_arrival_profile import weekly_rates  # noqa: E402
from loan_model.calendar_time import WorkingClock, EPOCH  # noqa: E402


# --------------------------------------------------------------------- loading
def load_log(path, case_attr_keys):
    """Events (case, activity, lifecycle, resource, time [UTC]) and case attributes."""
    ev_rows, case_rows = [], []
    for _, tr in etree.iterparse(path, tag='{*}trace'):
        attrs, events = {}, []
        for c in tr:
            tag = c.tag.split('}')[-1]
            if tag == 'event':
                a = {d.get('key'): d.get('value') for d in c}
                events.append(a)
            else:
                attrs[c.get('key')] = c.get('value')
        cid = attrs.get('concept:name')
        case_rows.append({'case': cid, **{k: attrs.get(k) for k in case_attr_keys}})
        for a in events:
            ev_rows.append((cid, a.get('concept:name'), (a.get('lifecycle:transition') or 'complete').lower(),
                            a.get('org:resource'), a.get('time:timestamp')))
        tr.clear()
        while tr.getprevious() is not None:
            del tr.getparent()[0]
    ev = pd.DataFrame(ev_rows, columns=['case', 'activity', 'lifecycle', 'resource', 'time'])
    ev['time'] = pd.to_datetime(ev['time'], utc=True, format='ISO8601')
    ev = ev.sort_values(['case', 'time'], kind='stable').reset_index(drop=True)
    return ev, pd.DataFrame(case_rows)


# ------------------------------------------------------------------ utilities
def quantiles(x, nq):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return None
    return [round(float(v), 1) for v in np.quantile(x, np.linspace(0, 1, nq))]


def shares(counter):
    tot = sum(counter.values())
    return {str(k): round(v / tot, 6) for k, v in sorted(counter.items(), key=lambda kv: -kv[1])} if tot else {}


def sessions_of(rows, lc_open, lc_close, system):
    """Active work sessions per activity: list of (activity, resource, start_s, end_s).
    A session opens at an open lifecycle and closes at the next close lifecycle of the
    same activity; sessions by system resources or without a recorded resource are
    dropped."""
    out, open_ = [], {}
    for act, lc, res, t in rows:
        if lc in lc_open:
            open_[act] = (res, t)
        elif lc in lc_close and act in open_:
            r, t0 = open_.pop(act)
            if r is not None and r not in system and t > t0:
                out.append((act, r, t0, t))
    return out


# ---------------------------------------------------------------------- mining
def mine(m, ev, cases):
    tz = m['timezone']
    system = set(m['system_resources'])
    lc_open, lc_close = set(m['lifecycle']['open']), set(m['lifecycle']['close'])
    entry = m['stage_entry_milestones']
    end_stages = set(m['end_stages'])
    stage_item = m['stage_work_items']
    item_stage = {w: s for s, w in stage_item.items()}
    optional = m['optional_work_items']
    ignored = set(m.get('ignored_activities', []))
    markers = m['markers']
    marker_of = defaultdict(set)     # activity -> marker names (one activity can carry several)
    for k, acts in markers.items():
        for a in acts:
            marker_of[a].add(k)
    timers = m['timers']
    buckets = m['visit_buckets']
    tol = m['session_attach_tolerance_s']
    nq = m['n_quantiles']
    dq = m['deadline_quantile']
    bucket = lambda v: str(min(v, buckets[-1]))

    ev = ev[~ev['activity'].isin(ignored)].copy()
    ev['t'] = (ev['time'] - ev['time'].min()).dt.total_seconds()
    t0_abs = ev['time'].min()

    # accumulators
    exits_by_visit = defaultdict(lambda: defaultdict(Counter))
    exits_by_flag = defaultdict(lambda: defaultdict(Counter))
    trans = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))   # s -> n -> field -> values
    n_sess = defaultdict(lambda: defaultdict(Counter))
    wpairs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))   # s -> n -> field -> (t_from, t_to)
    opt_hits = defaultdict(Counter)       # stage -> item -> visits with it
    stage_visits = Counter()
    item_sessions = defaultdict(list)     # item -> durations
    item_sess_per_visit = defaultdict(list)
    worker_sessions = defaultdict(lambda: defaultdict(list))
    worker_days = defaultdict(set)
    session_starts_local = []
    exit_age = defaultdict(list)
    timer_delays = defaultdict(list)
    timer_target_exits = Counter()
    flag_counts = Counter()
    offers_case, offers_visit = [], defaultdict(list)
    online_only = Counter()
    cycle, outcome, rounds, incomplete_cases = [], Counter(), Counter(), 0
    case_span = []

    for cid, g in ev.groupby('case', sort=False):
        acts, lcs, ress, ts = g['activity'].tolist(), g['lifecycle'].tolist(), g['resource'].tolist(), g['t'].tolist()
        # flags
        flags = set()
        for fname, spec in m['case_flags'].items():
            for a, r in zip(acts, ress):
                if a == spec['activity']:
                    if r in spec['resource_in']:
                        flags.add(fname)
                    break
        for f in flags:
            flag_counts[f] += 1
        # stage visits from milestones (consecutive repeats collapsed)
        ms = []
        for a, lc, r, t in zip(acts, lcs, ress, ts):
            # an entry key is either "ACTIVITY" (any lifecycle) or "ACTIVITY@lifecycle",
            # e.g. BPI2012 marks validation only by "W_Valideren aanvraag@schedule"
            st_ = entry.get(f'{a}@{lc}', entry.get(a))
            if st_ is not None and (not ms or ms[-1][0] != st_):
                ms.append((st_, t, r))
        if not ms:
            continue
        case_span.append((ts[0], ts[-1]))
        cycle.append(ts[-1] - ts[0])
        last_stage = ms[-1][0]
        outcome[last_stage if last_stage in end_stages else 'open'] += 1
        visits, vcount = [], Counter()
        for i, (s, t, r) in enumerate(ms):
            if s in end_stages:
                break
            vcount[s] += 1
            nxt = ms[i + 1] if i + 1 < len(ms) else None
            visits.append(dict(stage=s, entry=t, exit=nxt[1] if nxt else None, next=nxt[0] if nxt else None,
                               exit_res=nxt[2] if nxt else None, v=vcount[s], sessions=[], opt=set(), offers=0))
        rounds[vcount.get('validation', 0)] += 1
        incomplete_cases += vcount.get('incomplete', 0) > 0
        # sessions -> visits
        for act, r, s0, s1 in sessions_of(zip(acts, lcs, ress, ts), lc_open, lc_close, system):
            worker_sessions[r][act].append(s1 - s0)
            item_sessions[act].append(s1 - s0)
            day = (t0_abs + pd.Timedelta(seconds=s0)).tz_convert(tz)
            worker_days[r].add(day.date())
            session_starts_local.append(day)
            target_stage = item_stage.get(act) or optional.get(act, {}).get('stage')
            if target_stage is None:
                continue
            for vis in visits:
                if vis['stage'] == target_stage and vis['entry'] - tol <= s0 and (vis['exit'] is None or s0 < vis['exit']):
                    if act in optional:
                        vis['opt'].add(act)
                    else:
                        vis['sessions'].append((s0, s1))
                    break
        # offers
        n_off = 0
        last_anchor = {}
        for a, t in zip(acts, ts):
            ks = marker_of.get(a, ())
            if 'offer_created' in ks:
                n_off += 1
                for vis in visits:
                    if vis['entry'] <= t and (vis['exit'] is None or t < vis['exit']):
                        vis['offers'] += 1
                        break
            if 'offer_sent' in ks:
                online_only['online_only' if a in markers.get('offer_sent_online_only', []) else 'mail_and_online'] += 1
        offers_case.append(n_off)
        # visit statistics
        for vis in visits:
            s, n = vis['stage'], vis['next']
            stage_visits[s] += 1
            offers_visit[s].append(vis['offers'])
            if n is None:
                continue
            exit_age[s].append(vis['exit'] - ts[0])
            exits_by_visit[s][bucket(vis['v'])][n] += 1
            for item in vis['opt']:
                opt_hits[s][item] += 1
                exits_by_flag[s][optional[item]['flag']][n] += 1
            sess = sorted(vis['sessions'])
            n_sess[s][n][len(sess)] += 1
            item_sess_per_visit[stage_item[s]].append(len(sess))
            tr_ = trans[s][n]
            tr_['visit_duration'].append(vis['exit'] - vis['entry'])
            if sess:
                tr_['initial_delay'].append(max(0.0, sess[0][0] - vis['entry']))
                tr_['tail'].append(max(0.0, vis['exit'] - sess[-1][1]))
                wpairs[s][n]['initial_delay'].append((vis['entry'], max(vis['entry'], sess[0][0])))
                wpairs[s][n]['tail'].append((sess[-1][1], max(sess[-1][1], vis['exit'])))
                for (a0, a1), (b0, b1) in zip(sess, sess[1:]):
                    tr_['gap'].append(max(0.0, b0 - a1))
                    wpairs[s][n]['gap'].append((a1, max(a1, b0)))
            else:
                tr_['visit_duration_no_session'].append(vis['exit'] - vis['entry'])
            # timers
            for tm in timers:
                if s == tm['stage'] and n == tm['target']:
                    timer_target_exits[tm['name']] += 1
                    if vis['exit_res'] in tm.get('target_resource_in', [vis['exit_res']]):
                        anchors = [t for a, t in zip(acts, ts) if tm['anchor_marker'] in marker_of.get(a, ()) and t <= vis['exit']]
                        if anchors:
                            timer_delays[tm['name']].append(vis['exit'] - (anchors[-1] if tm['anchor'] == 'last' else anchors[0]))

    n_cases = len(cycle)
    # ---------------------------------------------------------------- assemble
    starts = pd.Series(session_starts_local)
    hrs = starts.dt.hour + starts.dt.minute / 60
    calendar = {
        'days': sorted(int(d) for d, sh in starts.dt.weekday.value_counts(normalize=True).items() if sh >= 0.01),
        'hour_min': int(np.floor(np.quantile(hrs, 0.01))),
        'hour_max': int(np.ceil(np.quantile(hrs, 0.99))),
        'weekend_session_share': round(float((starts.dt.weekday >= 5).mean()), 4),
        # share of work sessions per weekday (Mon..Sun): scales each worker's daily
        # presence, since active_day_share is measured over Mon-Fri
        'session_share_by_weekday': [round(float((starts.dt.weekday == d).mean()), 4) for d in range(7)],
        'timezone': tz,
    }

    # Waits that end in a session (or at a worker-logged exit) in working time, see
    # loan_model/calendar_time.py. Fields get a "_w" suffix (working seconds).
    clock = WorkingClock(calendar['days'], calendar['hour_min'], calendar['hour_max'])
    t0_local = t0_abs.tz_convert(tz)

    def working_durations(pairs):
        if not pairs:
            return []
        arr = np.array(pairs, dtype=float)
        loc = (pd.DatetimeIndex(t0_local + pd.to_timedelta(arr.ravel(), unit='s')).tz_localize(None)
               - pd.Timestamp(EPOCH)).total_seconds().to_numpy().reshape(arr.shape)
        w = clock.to_working(loc)
        return np.maximum(0.0, w[:, 1] - w[:, 0])

    stages = {}
    for s in stage_item:
        stages[s] = {
            'work_item': stage_item[s],
            'visits': stage_visits[s],
            'exits_by_visit': {b: shares(c) for b, c in sorted(exits_by_visit[s].items())},
            'exits_by_flag': {f: shares(c) for f, c in exits_by_flag[s].items()},
            'optional_work_items': {item: {'p_per_visit': round(k / stage_visits[s], 6), 'flag': optional[item]['flag']}
                                    for item, k in opt_hits[s].items()},
            'transitions': {
                n: {'n': len(f['visit_duration']), 'n_sessions': shares(n_sess[s][n]),
                    **{k: quantiles(v, nq) for k, v in f.items()},
                    **{k + '_w': quantiles(working_durations(v), nq) for k, v in wpairs[s][n].items()}}
                for n, f in trans[s].items()},
        }
    work_items = {}
    workers = {}
    min_n = m['min_worker_sessions']
    weekdays_span = pd.bdate_range(t0_abs.tz_convert(tz).date(), ev['time'].max().tz_convert(tz).date())
    for item in list(stage_item.values()) + list(optional):
        qualified = sorted(r for r, d in worker_sessions.items() if len(d.get(item, [])) >= min_n)
        work_items[item] = {'session_duration': quantiles(item_sessions[item], nq),
                            'n_sessions_total': len(item_sessions[item]),
                            'sessions_per_visit_mean': round(float(np.mean(item_sess_per_visit[item])), 3)
                            if item_sess_per_visit[item] else None,
                            'qualified_workers': qualified}
    for r, d in sorted(worker_sessions.items()):
        items = {item: {'n': len(v), 'session_duration': quantiles(v, nq)} for item, v in d.items() if len(v) >= min_n}
        if items:
            workers[r] = {'active_day_share': round(len(worker_days[r]) / len(weekdays_span), 4), 'sessions': items}
    arrivals_local = sorted((t0_abs + pd.Timedelta(seconds=round(a))).tz_convert(tz).tz_localize(None).to_pydatetime()
                            for a, _ in case_span)
    rates = weekly_rates(arrivals_local)
    span_days = (arrivals_local[-1] - arrivals_local[0]).total_seconds() / 86400
    # open cases: daily sample at noon local time, full months away from the log edges
    st = np.array([a for a, _ in case_span]); en = np.array([b for _, b in case_span])
    grid = np.arange(st.min() + 45 * 86400, en.max() - 45 * 86400, 86400)
    wip = [int(((st <= g) & (en > g)).sum()) for g in grid]
    attrs = {}
    for k in m['case_attributes']:
        col = cases[k]
        num = pd.to_numeric(col, errors='coerce')
        attrs[k] = {'quantiles': quantiles(num.dropna(), nq)} if num.notna().mean() > 0.95 else {'shares': shares(Counter(col))}
    model = {
        'meta': {'log': m['log_name'], 'activity_map': m['_source_map'], 'timezone': tz,
                 'first_event': str(ev['time'].min()), 'last_event': str(ev['time'].max()), 'n_cases': n_cases,
                 'mined_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                 'units': 'seconds', 'n_quantiles': nq},
        'start_stage': m['start_stage'],
        'end_stages': m['end_stages'],
        'arrivals': {'rates_per_hour_of_week': [round(r, 6) for r in rates], 'per_day': round(n_cases / span_days, 3)},
        'case_flags': {f: round(k / n_cases, 6) for f, k in flag_counts.items()},
        'case_attributes': attrs,
        'offers': {'per_case': shares(Counter(offers_case)),
                   'per_visit_mean': {s: round(float(np.mean(v)), 4) for s, v in offers_visit.items()},
                   'sent_channel': shares(online_only)},
        'stages': stages,
        'work_items': work_items,
        'workers': workers,
        'calendar': calendar,
        'timers': {tm['name']: {'stage': tm['stage'], 'target': tm['target'], 'anchor_marker': tm['anchor_marker'],
                                'anchor': tm['anchor'], 'delay': quantiles(timer_delays[tm['name']], nq),
                                'n': len(timer_delays[tm['name']]),
                                'share_of_target_exits': round(len(timer_delays[tm['name']]) / max(1, timer_target_exits[tm['name']]), 4)}
                   for tm in timers},
        'deadlines': {s: round(float(np.quantile(v, dq)), 1) for s, v in exit_age.items()},
        'deadline_quantile': dq,
        'validation_targets': {
            'cycle_time': {f'p{p}': round(float(np.quantile(cycle, p / 100)), 1) for p in (10, 25, 50, 75, 90, 95)},
            'cycle_time_quantiles': quantiles(cycle, nq),
            'stage_visit_duration': {s: quantiles([d for f in trans[s].values() for d in f['visit_duration']], nq)
                                     for s in stage_item},
            'outcome': shares(outcome),
            'incomplete_case_share': round(incomplete_cases / n_cases, 4),
            'validation_rounds': shares(rounds),
            'offers_per_case': shares(Counter(offers_case)),
            'open_cases': {'median': int(np.median(wip)), 'p10': int(np.quantile(wip, 0.1)), 'p90': int(np.quantile(wip, 0.9))},
            'arrivals_per_day': round(n_cases / span_days, 3),
            'human_workers': len(workers),
        },
    }
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--map', required=True, help='activity map JSON')
    ap.add_argument('--out', required=True, help='stage model JSON to write')
    ap.add_argument('--log', help='event log path (default: log_path in the map, relative to the repo root)')
    args = ap.parse_args()
    with open(args.map) as f:
        m = json.load(f)
    m['_source_map'] = os.path.basename(args.map)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(args.map), '..', '..'))
    log = args.log or os.path.join(repo_root, m['log_path'])
    print(f'Loading {log} ...', flush=True)
    ev, cases = load_log(log, m['case_attributes'])
    print(f'{ev["case"].nunique()} cases, {len(ev)} events', flush=True)
    model = mine(m, ev, cases)
    with open(args.out, 'w') as f:
        json.dump(model, f, indent=1)
    vt = model['validation_targets']
    print(f"Wrote {args.out}: {len(model['stages'])} stages, {len(model['workers'])} workers, "
          f"cycle time p50 {vt['cycle_time']['p50'] / 86400:.1f} d, open cases {vt['open_cases']['median']}", flush=True)


if __name__ == '__main__':
    main()
