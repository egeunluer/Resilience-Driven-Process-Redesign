#!/usr/bin/env python3
"""
Module 1 — Empirical Pre-Processing

Given a raw event log, produces:
  - resource_processing_times[resource][activity]: capped-median processing time in seconds
  - resource_tier_map[resource][activity]: performance tier label (0=slow, ..., n-1=fast)

Output is a single JSON artifact consumed by Modules 2, 3, and 4.

Usage:
  python empirical_preprocessing.py LOG_PATH OUTPUT_PATH [options]

Examples:
  python empirical_preprocessing.py "../Event Logs/BPI_Challenge_2012.xes" \
      artifacts/bpi12w.json --activity-prefix W_ --n-tiers 3 --n-min 3

  python empirical_preprocessing.py "../Event Logs/BPI Challenge 2017.xes" \
      artifacts/bpi2017.json --activity-prefix W_ --n-tiers 3 --n-min 3
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pm4py

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

TIER_LABELS = {0: "slow", 1: "mid", 2: "fast", 3: "tier_3", 4: "tier_4"}


# ---------------------------------------------------------------------------
# Step 1: Load and normalise the log
# ---------------------------------------------------------------------------

def load_and_prepare(log_path: str, activity_prefix: str | None) -> tuple[pd.DataFrame, bool]:
    """
    Load XES or CSV, normalise column names, optionally filter by activity prefix.

    For CSV, column names are detected automatically using common aliases.
    Required columns after normalisation: case_id, activity, resource, timestamp.

    Returns (df, has_true_pt) where:
      - df has columns: case_id, activity, resource, timestamp, lifecycle
      - has_true_pt = True when both 'start' and 'complete' lifecycle values exist
        (enables direct start→complete processing-time computation)
    """
    path = Path(log_path)
    log.info(f"Reading {path.name} …")

    if path.suffix.lower() == ".xes":
        df = pm4py.read_xes(str(path))
        rename = {
            "case:concept:name": "case_id",
            "concept:name": "activity",
            "org:resource": "resource",
            "time:timestamp": "timestamp",
        }
        df = df.rename(columns=rename)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(str(path))
        df = _normalise_csv_columns(df)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}. Supported: .xes, .csv")

    # Detect lifecycle column (handles 'lifecycle:transition' and variants)
    lifecycle_col = next(
        (c for c in df.columns if "lifecycle" in c.lower() and c != "case:lifecycle:transition"),
        None,
    )
    if lifecycle_col:
        df = df.rename(columns={lifecycle_col: "lifecycle"})
        lc_lower = df["lifecycle"].dropna().str.lower()
        # Only treat as true-PT mode if at least one activity has BOTH start and complete events.
        # This avoids false positives from logs where "Start" appears only as an artificial
        # trace-start marker activity (e.g. helpdesk CSV), not as a real lifecycle transition.
        if "start" in set(lc_lower) and "complete" in set(lc_lower):
            start_acts = set(df[lc_lower == "start"]["activity"].dropna().unique())
            complete_acts = set(df[lc_lower == "complete"]["activity"].dropna().unique())
            has_true_pt = len(start_acts & complete_acts) > 0
        else:
            has_true_pt = False
    else:
        df["lifecycle"] = "complete"
        has_true_pt = False

    # Filter to chosen activity prefix
    if activity_prefix:
        before = len(df)
        df = df[df["activity"].str.startswith(activity_prefix)]
        log.info(f"Activity prefix filter '{activity_prefix}': {before} → {len(df)} events")

    # Drop rows without a resource (automated system events)
    df = df.dropna(subset=["resource"])

    # Normalise timestamps to tz-naive for arithmetic
    if pd.api.types.is_object_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    if hasattr(df["timestamp"].dtype, "tz") and df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    df = df[["case_id", "activity", "resource", "timestamp", "lifecycle"]].copy()
    df = df.sort_values(["case_id", "timestamp"]).reset_index(drop=True)

    log.info(
        f"Prepared: {len(df)} events | "
        f"{df['activity'].nunique()} activities | "
        f"{df['resource'].nunique()} resources | "
        f"mode={'true_pt' if has_true_pt else 'sojourn_approx'}"
    )
    return df, has_true_pt


# Common CSV column name aliases → canonical name
_CSV_ALIASES: dict[str, list[str]] = {
    "case_id": [
        "case id", "case:concept:name", "caseid", "case_id", "traceid", "trace id",
        "id_case", "case", "case no", "case no.",
    ],
    "activity": [
        "activity", "concept:name", "task", "event", "event name", "activity name",
        "activityname", "action",
    ],
    "resource": [
        "resource", "org:resource", "user", "agent", "employee", "performer",
        "responsible", "originator",
    ],
    "timestamp": [
        "complete timestamp", "time:timestamp", "timestamp", "end timestamp",
        "end time", "complete time", "datetime", "date", "event time",
        "start timestamp", "start time",
    ],
}


def _normalise_csv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map CSV columns to canonical names using alias lookup (case-insensitive)."""
    lower_cols = {c.lower().strip(): c for c in df.columns}
    rename = {}
    missing = []
    for canonical, aliases in _CSV_ALIASES.items():
        if canonical in df.columns:
            continue  # already named correctly
        matched = next((lower_cols[a] for a in aliases if a in lower_cols), None)
        if matched:
            rename[matched] = canonical
        else:
            missing.append(canonical)

    if rename:
        log.info(f"CSV column mapping: {rename}")
    df = df.rename(columns=rename)

    # Report any columns that couldn't be mapped
    still_missing = [c for c in missing if c not in df.columns]
    if still_missing:
        raise ValueError(
            f"Could not find columns for: {still_missing}. "
            f"CSV has columns: {list(df.columns)}. "
            f"Add an alias or rename the column."
        )
    return df


# ---------------------------------------------------------------------------
# Step 2a: True processing times (start → complete pairs)
# ---------------------------------------------------------------------------

def _compute_true_pt(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Match start/complete pairs per (case_id, activity, resource) by sorted order.
    Vectorised: O(n log n) via groupby cumcount + merge.
    """
    starts = df[df["lifecycle"].str.lower() == "start"][
        ["case_id", "activity", "resource", "timestamp"]
    ].copy()
    completes = df[df["lifecycle"].str.lower() == "complete"][
        ["case_id", "activity", "resource", "timestamp"]
    ].copy()

    key = ["case_id", "activity", "resource"]
    starts = starts.sort_values(key + ["timestamp"])
    completes = completes.sort_values(key + ["timestamp"])
    starts["_pair"] = starts.groupby(key).cumcount()
    completes["_pair"] = completes.groupby(key).cumcount()

    merged = starts.merge(
        completes, on=key + ["_pair"], suffixes=("_s", "_c")
    )
    merged["pt"] = (merged["timestamp_c"] - merged["timestamp_s"]).dt.total_seconds()

    excluded = []
    bad = merged[merged["pt"] <= 0]
    for _, row in bad.iterrows():
        excluded.append({
            "resource": row["resource"],
            "activity": row["activity"],
            "reason": "zero_or_negative_true_pt",
        })

    records = merged[merged["pt"] > 0][["resource", "activity", "pt"]]
    return records, excluded


# ---------------------------------------------------------------------------
# Step 2b: Sojourn approximation (COMPLETE-only logs)
# ---------------------------------------------------------------------------

def _compute_sojourn_pt(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Approximate processing time as (this event timestamp − previous event
    timestamp within the same case). The first event in each case is excluded
    because it has no predecessor.

    Produces sojourn time (waiting + working), not true working time.
    This limitation is recorded in the output artifact's metadata.
    """
    comp = df[df["lifecycle"].str.lower() == "complete"].copy()
    comp = comp.sort_values(["case_id", "timestamp"])
    comp["prev_ts"] = comp.groupby("case_id")["timestamp"].shift(1)
    comp["pt"] = (comp["timestamp"] - comp["prev_ts"]).dt.total_seconds()

    excluded = []
    bad = comp[comp["pt"].notna() & (comp["pt"] <= 0)]
    for _, row in bad.iterrows():
        excluded.append({
            "resource": row["resource"],
            "activity": row["activity"],
            "reason": "zero_or_negative_sojourn",
        })

    # Drop first events in each case (no predecessor) and bad values
    records = comp.dropna(subset=["prev_ts"])
    records = records[records["pt"] > 0][["resource", "activity", "pt"]]
    return records, excluded


# ---------------------------------------------------------------------------
# Step 3: Aggregate to median, apply cap and min-support filter
# ---------------------------------------------------------------------------

def aggregate(
    records: pd.DataFrame,
    n_min: int,
    cap_pct: float = 95.0,
) -> tuple[dict, dict, list]:
    """
    For each (resource, activity) pair:
      1. Cap values above the cap_pct-th percentile.
      2. Compute the median.
      3. Exclude pairs with support < n_min or median ≤ 0.

    Fallback: if all resources for an activity are excluded by the n_min filter,
    re-admit the resource with the highest support for that activity (with a
    warning). This prevents simulator deadlock on rare activities.

    Returns:
      resource_processing_times[resource][activity] = float (seconds)
      support_counts[resource][activity] = int
      excluded_pairs list
    """
    excluded = []
    resource_pt: dict[str, dict[str, float]] = {}
    support_counts: dict[str, dict[str, int]] = {}
    # Track excluded candidates per activity for the fallback
    excluded_by_activity: dict[str, list[dict]] = {}

    for (resource, activity), grp in records.groupby(["resource", "activity"]):
        pts = grp["pt"].values
        n = len(pts)
        cap = float(np.percentile(pts, cap_pct))
        pts_capped = np.minimum(pts, cap)
        median_pt = float(np.median(pts_capped))

        if n < n_min:
            reason = f"support_{n}_below_n_min_{n_min}"
        elif median_pt <= 0:
            reason = "median_nonpositive_after_cap"
        else:
            reason = None

        if reason:
            excluded.append({"resource": resource, "activity": activity, "reason": reason})
            excluded_by_activity.setdefault(activity, []).append({
                "resource": resource, "n": n, "median_pt": median_pt,
            })
            continue

        resource_pt.setdefault(resource, {})[activity] = median_pt
        support_counts.setdefault(resource, {})[activity] = n

    # Fallback: re-admit best resource for activities with zero eligible resources
    admitted_by_activity = {
        activity
        for resource, acts in resource_pt.items()
        for activity in acts
    }
    for activity, candidates in excluded_by_activity.items():
        if activity not in admitted_by_activity:
            # Pick the candidate with the highest support and a positive median
            valid_candidates = [c for c in candidates if c["median_pt"] > 0]
            if not valid_candidates:
                continue  # truly hopeless; validation will raise
            best = max(valid_candidates, key=lambda c: c["n"])
            log.warning(
                f"FALLBACK: activity '{activity}' had 0 eligible resources after n_min={n_min} "
                f"filter — re-admitting '{best['resource']}' (support={best['n']}, "
                f"median_pt={best['median_pt']:.0f}s). "
                f"Use this pair's estimates with caution."
            )
            # Re-compute properly for this resource
            grp = records[(records["resource"] == best["resource"]) & (records["activity"] == activity)]
            pts = grp["pt"].values
            cap = float(np.percentile(pts, cap_pct))
            pts_capped = np.minimum(pts, cap)
            median_pt = float(np.median(pts_capped))
            resource_pt.setdefault(best["resource"], {})[activity] = median_pt
            support_counts.setdefault(best["resource"], {})[activity] = best["n"]
            # Mark in excluded list as re-admitted
            for exc in excluded:
                if exc["resource"] == best["resource"] and exc["activity"] == activity:
                    exc["reason"] += "_FALLBACK_READMITTED"

    return resource_pt, support_counts, excluded


# ---------------------------------------------------------------------------
# Step 4: Performance tiering
# ---------------------------------------------------------------------------

def compute_tiers(resource_pt: dict, n_tiers: int) -> tuple[dict, dict]:
    """
    For each activity, quantile-bin its eligible resources into n_tiers tiers.
    Tier 0 = slowest, tier (n_tiers-1) = fastest.

    Uses rank-based assignment so ties are handled gracefully:
      tier = floor(rank_from_fast * effective_tiers / n_eligible)
    clamped to [0, effective_tiers-1].
    """
    # Invert: activity → {resource: median_pt}
    activity_res: dict[str, dict[str, float]] = {}
    for resource, acts in resource_pt.items():
        for activity, pt in acts.items():
            activity_res.setdefault(activity, {})[resource] = pt

    resource_tier_map: dict[str, dict[str, int]] = {}
    activity_tiers: dict[str, dict] = {}

    for activity, res_pt_map in activity_res.items():
        n_eligible = len(res_pt_map)
        effective = min(n_tiers, n_eligible)
        if effective < n_tiers:
            log.warning(
                f"'{activity}': clamped tiers {n_tiers} → {effective} "
                f"(only {n_eligible} eligible resources)"
            )

        # Sort ascending by pt: index 0 = fastest (lowest pt), index n-1 = slowest.
        # Tier 0 = slow (highest pt), tier effective-1 = fast (lowest pt).
        # We sort ascending so that index 0 = fastest → maps to tier effective-1.
        sorted_res = sorted(res_pt_map.items(), key=lambda x: x[1])  # ascending: fast first
        n = len(sorted_res)

        tier_assignments: dict[str, int] = {}
        for rank_fast, (resource, _) in enumerate(sorted_res):
            # rank_fast=0 → fastest → tier effective-1; rank_fast=n-1 → slowest → tier 0
            rank_slow = n - 1 - rank_fast
            tier = min(int(rank_slow * effective / n), effective - 1)
            tier_assignments[resource] = tier

        # Build summary
        tier_groups: dict[int, list[str]] = {}
        for resource, tier in tier_assignments.items():
            tier_groups.setdefault(tier, []).append(resource)

        activity_tiers[activity] = {}
        for tier_id in sorted(tier_groups):
            members = tier_groups[tier_id]
            mean_pt = float(np.mean([res_pt_map[r] for r in members]))
            activity_tiers[activity][str(tier_id)] = {
                "label": TIER_LABELS.get(tier_id, f"tier_{tier_id}"),
                "members": sorted(members),
                "mean_processing_time_s": mean_pt,
            }

        for resource, tier in tier_assignments.items():
            resource_tier_map.setdefault(resource, {})[activity] = tier

    return resource_tier_map, activity_tiers


# ---------------------------------------------------------------------------
# Step 5: Validation
# ---------------------------------------------------------------------------

def validate(resource_pt: dict, activity_tiers: dict, all_activities: set) -> list[str]:
    """
    Check that every log activity has at least one eligible resource.
    Activities with zero eligible resources are collected and returned as a
    'deadlock-risk' list — they appear in the artifact and must be removed
    from the Petri net or given a manual fallback before simulation.

    Logs tier composition for all eligible activities.
    """
    at_risk = []
    for activity in sorted(all_activities):
        has_any = any(activity in acts for acts in resource_pt.values())
        if not has_any:
            log.warning(
                f"DEADLOCK RISK: activity '{activity}' has NO eligible resources after "
                "all exclusions (possibly only SCHEDULE/non-start-complete lifecycle events "
                "in the log). Exclude this activity from the Petri net before simulation."
            )
            at_risk.append(activity)

    log.info("Tier composition:")
    for activity, tiers in sorted(activity_tiers.items()):
        for tid, info in sorted(tiers.items()):
            log.info(
                f"  {activity:40s} | tier {tid} ({info['label']:5s}): "
                f"{len(info['members']):3d} members, "
                f"mean={info['mean_processing_time_s']:.0f}s"
            )
    return at_risk


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    log_path: str,
    output_path: str,
    activity_prefix: str | None = None,
    n_tiers: int = 3,
    n_min: int = 3,
    cap_pct: float = 95.0,
) -> dict:
    df, has_true_pt = load_and_prepare(log_path, activity_prefix)
    all_activities = set(df["activity"].unique())
    mode = "true" if has_true_pt else "sojourn_approximation"

    log.info(f"Computing processing times (mode={mode}) …")
    if has_true_pt:
        records, excluded_stage1 = _compute_true_pt(df)
    else:
        records, excluded_stage1 = _compute_sojourn_pt(df)
    log.info(f"  Raw (resource, activity, pt) records: {len(records)}")

    if excluded_stage1:
        log.warning(f"  Stage-1 exclusions (zero/negative/unmatched): {len(excluded_stage1)}")

    log.info(f"Aggregating (cap={cap_pct}th-pct, n_min={n_min}) …")
    resource_pt, support_counts, excluded_stage2 = aggregate(records, n_min, cap_pct)
    if excluded_stage2:
        log.warning(f"  Stage-2 exclusions (low support or nonpositive median): {len(excluded_stage2)}")
        for exc in excluded_stage2:
            log.warning(f"    {exc}")

    log.info(f"Computing tiers (n_tiers={n_tiers}) …")
    resource_tier_map, activity_tiers = compute_tiers(resource_pt, n_tiers)

    at_risk = validate(resource_pt, activity_tiers, all_activities)

    artifact = {
        "activity_tiers": activity_tiers,
        "resource_tier_map": resource_tier_map,
        "resource_processing_times": resource_pt,
        "support_counts": support_counts,
        "excluded_pairs": excluded_stage1 + excluded_stage2,
        "activities_at_risk": at_risk,
        "config_used": {
            "n_tiers": n_tiers,
            "n_min_support": n_min,
            "cap_percentile": cap_pct,
            "tiering_method": "quantile",
            "processing_time_mode": mode,
            "activity_prefix_filter": activity_prefix,
            "log_path": str(log_path),
        },
        "summary": {
            "n_resources": len(resource_pt),
            "n_activities": len(activity_tiers),
            "n_eligible_pairs": sum(len(acts) for acts in resource_pt.values()),
            "n_excluded_pairs": len(excluded_stage1) + len(excluded_stage2),
            "n_activities_at_risk": len(at_risk),
        },
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(artifact, f, indent=2)
    log.info(f"Artifact written → {out}")
    log.info(
        f"Summary: {artifact['summary']['n_resources']} resources, "
        f"{artifact['summary']['n_activities']} activities, "
        f"{artifact['summary']['n_eligible_pairs']} eligible pairs, "
        f"{artifact['summary']['n_excluded_pairs']} excluded"
    )
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Module 1 — Empirical Pre-Processing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("log_path", help="Path to XES event log")
    parser.add_argument("output_path", help="Path for output JSON artifact")
    parser.add_argument(
        "--activity-prefix",
        default=None,
        help="Keep only activities whose name starts with this prefix (e.g. 'W_')",
    )
    parser.add_argument("--n-tiers", type=int, default=3, help="Number of performance tiers")
    parser.add_argument(
        "--n-min",
        type=int,
        default=3,
        help="Minimum support count; pairs below this are excluded (D4 filter)",
    )
    parser.add_argument(
        "--cap-pct",
        type=float,
        default=95.0,
        help="Percentile for outlier capping before median computation",
    )
    args = parser.parse_args()

    try:
        run(
            args.log_path,
            args.output_path,
            activity_prefix=args.activity_prefix,
            n_tiers=args.n_tiers,
            n_min=args.n_min,
            cap_pct=args.cap_pct,
        )
    except Exception as e:
        log.error(str(e))
        sys.exit(1)
