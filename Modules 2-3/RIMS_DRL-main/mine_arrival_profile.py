#!/usr/bin/env python3
"""
Mine an empirical weekly case-arrival profile for the RIMS simulator.

Replaces the 'custom' (Prophet) arrival model where that model is unusable --
for Consulta, the shipped Prophet model predicts 0.001 arrivals/hour on its own
training period (actual: 0.269/hour), so the simulated rate came almost entirely
from noise plus np.ceil rounding (~12.4 cases/day against 6.44/day in the log).

For each of the 168 hours of the week (Monday 00:00 = slot 0), the rate is the
number of cases whose first event falls in that slot, divided by how many times
that slot occurs between the first and the last case arrival. InterTriggerTimer
('empirical_weekly' type) then draws a Poisson number of arrivals per simulated
hour from these rates, uniformly placed within the hour.

Hour of week is taken from the timestamps' wall-clock time exactly as written in
the XES (ignoring the UTC offset), the same clock the simulator applies resource
calendars to.

Usage:
  python3 mine_arrival_profile.py --dataset ConsultaDataMining201618 \\
      --inputs input_ConsultaDataMining201618.json input_ConsultaDataMining20161820.json
"""
import argparse
import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta


def case_arrivals(xes_path):
    """Wall-clock time of each case's first event (earliest by actual instant)."""
    arrivals = []
    for _, elem in ET.iterparse(xes_path, events=('end',)):
        if elem.tag.split('}')[-1] != 'trace':
            continue
        stamps = [d.get('value') for ev in elem if ev.tag.split('}')[-1] == 'event'
                  for d in ev if d.get('key') == 'time:timestamp']
        if stamps:
            first = min(stamps, key=lambda s: datetime.fromisoformat(s))
            arrivals.append(datetime.fromisoformat(first).replace(tzinfo=None))
        elem.clear()
    return sorted(arrivals)


def weekly_rates(arrivals):
    counts = [0] * 168
    for a in arrivals:
        counts[a.weekday() * 24 + a.hour] += 1
    exposure = [0] * 168
    t = arrivals[0].replace(minute=0, second=0, microsecond=0)
    while t <= arrivals[-1]:
        exposure[t.weekday() * 24 + t.hour] += 1
        t += timedelta(hours=1)
    return [c / e if e else 0.0 for c, e in zip(counts, exposure)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True, help="Folder name under example/")
    parser.add_argument('--inputs', nargs='+', required=True,
                        help="input_*.json file names (in example/<dataset>/) to write the profile into")
    args = parser.parse_args()

    folder = os.path.join('example', args.dataset)
    arrivals = case_arrivals(os.path.join(folder, f'{args.dataset}.xes'))
    rates = weekly_rates(arrivals)
    days = (arrivals[-1] - arrivals[0]).total_seconds() / 86400
    print(f"{len(arrivals)} cases over {days:.0f} days = {len(arrivals) / days:.2f}/day; "
          f"profile mean {sum(rates) / 7:.2f}/day")

    source = (f"Hour-of-week arrival rates (cases/hour, slot 0 = Monday 00:00 wall-clock as written "
              f"in the log) from {len(arrivals)} case arrivals in {args.dataset}.xes over {days:.0f} "
              f"days, written by mine_arrival_profile.py. Replaces the 'custom' Prophet model.")
    block = ('"interTriggerTimer": {\n        "type": "empirical_weekly",\n'
             f'        "rates_per_hour_of_week": [{", ".join(f"{r:.6f}" for r in rates)}],\n'
             f'        "source": {json.dumps(source)}\n    }}')
    for name in args.inputs:
        path = os.path.join(folder, name)
        with open(path) as f:
            text = f.read()
        # Text replacement, not a JSON re-dump: the input files are hand-formatted.
        text, n = re.subn(r'"interTriggerTimer":\s*\{[^{}]*\}', lambda m: block, text)
        if n != 1:
            raise ValueError(f"Expected exactly one flat \"interTriggerTimer\" block in {path}, found {n}")
        json.loads(text)
        with open(path, 'w') as f:
            f.write(text)
        print(f"Wrote empirical weekly arrival profile to {path}")


if __name__ == '__main__':
    main()
