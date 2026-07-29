#!/usr/bin/env python3
"""
Module 1 — P* Calibration

Computes the acceptable-performance threshold P* used by Module 3's Reward
Function 1 (`r(t) = [Phi(s_{t+1}) - Phi(s_t)] - c * max(0, P(t) - P*)`), where
P(t) is the process performance indicator (lead time of completed cases).

P* is defined here as a percentile (default: 95th) of the per-case lead time
distribution in the raw event log — not the raw historical maximum, which is
dominated by single outlier traces and would make the deviation penalty
never trigger. Uses the same log loading/filtering path as
empirical_preprocessing.py (load_and_prepare) so the case slice matches
exactly what Module 1's resource_processing_times/resource_tier_map were
computed from.

Usage:
  python compute_p_star.py LOG_PATH [options]

Examples:
  python compute_p_star.py "../Event Logs/BPI Challenge 2017.xes" \
      --activity-prefix W_ --percentile 95 --output artifacts/bpi2017_p_star.json

  python compute_p_star.py "../Event Logs/BPI_Challenge_2012.xes" \
      --activity-prefix W_ --percentile 95 --output artifacts/bpi12w_p_star.json

  python compute_p_star.py "../Event Logs/helpdesk_anonymized.csv" \
      --percentile 95 --output artifacts/helpdesk_p_star.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from empirical_preprocessing import load_and_prepare

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def compute_lead_times(log_path: str, activity_prefix: str | None) -> pd.Series:
    """
    Per-case lead time (seconds): max(timestamp) - min(timestamp) among the
    events that survive load_and_prepare's normalisation and activity-prefix
    filter. Matches the case slice Module 1 uses elsewhere.
    """
    df, _, _ = load_and_prepare(log_path, activity_prefix)
    span = df.groupby("case_id")["timestamp"].agg(lambda x: x.max() - x.min())
    return span.dt.total_seconds()


def summarize(lead_times_s: pd.Series, percentile: float) -> dict:
    p_star = float(lead_times_s.quantile(percentile / 100.0))
    worst_case_id = lead_times_s.idxmax()
    return {
        "percentile": percentile,
        "p_star_seconds": p_star,
        "p_star_hours": p_star / 3600.0,
        "p_star_days": p_star / 86400.0,
        "distribution_seconds": {
            "count": int(lead_times_s.count()),
            "min": float(lead_times_s.min()),
            "mean": float(lead_times_s.mean()),
            "median": float(lead_times_s.median()),
            "p90": float(lead_times_s.quantile(0.90)),
            "p95": float(lead_times_s.quantile(0.95)),
            "p99": float(lead_times_s.quantile(0.99)),
            "max": float(lead_times_s.max()),
        },
        "worst_case_id": str(worst_case_id),
    }


def run(
    log_path: str,
    activity_prefix: str | None = None,
    percentile: float = 95.0,
    output_path: str | None = None,
) -> dict:
    log.info(f"Computing per-case lead times from {log_path} (prefix={activity_prefix!r}) …")
    lead_times_s = compute_lead_times(log_path, activity_prefix)
    result = summarize(lead_times_s, percentile)
    result["config_used"] = {
        "log_path": str(log_path),
        "activity_prefix_filter": activity_prefix,
        "percentile": percentile,
    }

    d = result["distribution_seconds"]
    log.info(
        f"Lead time (h): median={d['median']/3600:.2f} mean={d['mean']/3600:.2f} "
        f"p90={d['p90']/3600:.2f} p95={d['p95']/3600:.2f} p99={d['p99']/3600:.2f} "
        f"max={d['max']/3600:.2f}"
    )
    log.info(
        f"P* (p{percentile}) = {result['p_star_seconds']:.1f} s "
        f"= {result['p_star_hours']:.2f} h = {result['p_star_days']:.2f} d"
    )
    log.info(f"Worst case_id: {result['worst_case_id']} (max={d['max']/3600:.2f} h)")

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        log.info(f"Written → {out}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Module 1 — P* calibration (percentile of per-case lead time)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("log_path", help="Path to XES or CSV event log")
    parser.add_argument(
        "--activity-prefix",
        default=None,
        help="Keep only activities whose name starts with this prefix (e.g. 'W_'). "
             "Use the same value as the corresponding empirical_preprocessing.py run.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=95.0,
        help="Percentile of the per-case lead time distribution to use as P*",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the result as a JSON artifact",
    )
    args = parser.parse_args()

    try:
        run(
            args.log_path,
            activity_prefix=args.activity_prefix,
            percentile=args.percentile,
            output_path=args.output,
        )
    except Exception as e:
        log.error(str(e))
        sys.exit(1)
