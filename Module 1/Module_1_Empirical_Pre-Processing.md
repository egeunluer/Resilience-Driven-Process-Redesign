# Module 1 – Empirical Pre-Processing: Implementation Specification

## Purpose

Given a raw event log, produce two empirical artifacts:

1. **Per-resource, per-activity processing times** — consumed by Module 2 for duration
   sampling when the agent assigns a specific resource to a specific activity.
2. **Per-resource, per-activity tier labels** — consumed by Module 3 at runtime to generate
   the cluster-layer observation in parallel with the agent's individual-layer decisions, and
   by Module 4 to translate the agent's individual-level actions and observations into
   interpretable cluster-level representations for policy distillation.

The agent (Module 3) operates entirely at the individual resource level — it assigns specific
resources to specific tasks. Module 1's tier labels are never shown to the agent. They exist
solely to support the dual-layer observation system in Module 3 and the distillation pipeline
in Module 4.

This module must run **once per event log** and must be re-runnable as a standalone
pre-processing step, independent of simulation or training code.

---

## 1. Input

- An event log in standard XES or CSV format, loadable via PM4Py (`pm4py.read_xes` /
  `pm4py.format_dataframe`).
- Required columns/attributes after loading into a `pandas.DataFrame` (PM4Py convention):
  - `case:concept:name` — case/trace ID
  - `concept:name` — activity label
  - `org:resource` — resource identifier
  - `time:timestamp` — event timestamp
  - Optional: `lifecycle:transition` (if present, filter to `"complete"` events only; if both
    `"start"` and `"complete"` exist for the same activity instance, processing time is
    `complete_timestamp - start_timestamp`; otherwise use the sojourn approximation
    described in Section 2.1).

---

## 2. Stage 1 — Processing Time Computation

**Goal:** compute a reliable per-resource, per-activity processing time estimate from the
event log. These values are the foundation for both Module 2's duration sampling and the
performance tiering in Stage 2.

### 2.1 Compute processing time per (resource, activity)
```
processing_time(resource, activity) = median over all instances of
    (complete_timestamp - start_timestamp)
```
- If only single timestamps exist per event (no start/complete pair), approximate processing
  time as the time delta between this event's timestamp and the **previous event's timestamp
  within the same case**. This must be implemented as a fallback branch, detected
  automatically from whether `lifecycle:transition` values include both `start` and
  `complete`. **Important:** this fallback produces a *sojourn time approximation* that
  includes waiting time, not just active working time. This limitation must be recorded in
  the output artifact's metadata.
- Before computing the median, apply outlier capping: truncate any value above the 95th
  percentile of that (resource, activity) pair to the 95th percentile value.
- Any (resource, activity) pair where the computed value is zero or negative is excluded
  from all downstream computation and logged as a data anomaly.

---

## 3. Stage 2 — Activity-Specific Performance Tiering

**Goal:** for each activity, independently cluster the resources that have performed it into
performance tiers based on their processing times. A resource receives a tier label per
activity it has performed. Resources that have never performed an activity receive no tier
label for it — they are simply ineligible for that activity, handled by action masking in
Module 3.

### 3.1 Cluster resources into tiers per activity
For each activity `a`:
- Collect all resources that have at least one valid processing time for `a`.
- Cluster their `processing_time(resource, a)` values into `t` tiers using **quantile
  binning** — resources are divided into equal-sized groups by ranked processing time.
- The effective number of tiers is clamped to `min(t_config, n_eligible_resources)`. If
  this cap is applied, log a warning.
- Tier labels are assigned ordinally: `0 = slow`, `1 = mid`, `2 = fast` (or up to
  `t_config - 1`). Higher label = faster.

### 3.2 Configuration
- `n_tiers` (default: 3) — number of performance tiers.

---

## 4. Output Contract

Persist as a structured artifact (JSON or pickle/parquet) with:

```json
{
  "activity_tiers": {
    "<activity_name>": {
      "<tier_id>": {
        "label": "slow | mid | fast",
        "members": ["resource_a", "resource_b", ...],
        "mean_processing_time": <float>
      }
    }
  },
  "resource_tier_map": {
    "<resource_id>": {
      "<activity_name>": <tier_id>,
      ...
    }
  },
  "resource_processing_times": {
    "<resource_id>": {
      "<activity_name>": <float>,
      ...
    }
  },
  "excluded_pairs": [
    {"resource": "<resource_id>", "activity": "<activity_name>", "reason": "..."}
  ],
  "activities_at_risk": ["<activity_name>", ...],
  "config_used": {
    "n_tiers": 3,
    "tiering_method": "quantile",
    "processing_time_mode": "true | sojourn_approximation"
  },
  "P_star": <float>,
  "normalization_cycle_time": <float>,
  "ppi_window_n": <int>
}
```

**Fields added after the original spec, not covered above:**
- **`activities_at_risk`** — activities that, after all exclusion filters (Section 5), would have
  zero eligible resources. Each gets a synthetic placeholder resource (below) instead of the hard
  error Section 5 originally specified.
- **`P_star`** — historically the p95 of this log's per-case lead-time distribution, kept in the
  artifact for reference. **This is not the value actually used for training or evaluation.** The
  operative `P*` — `2 × pooled median(cycle_times)` from seeded, no-shock, least-used-policy
  *simulation* baselines — is computed downstream, once per (dataset, resource pool) pair, by
  `calibrate_resilience_constants.py` (Modules 2-3), and persisted to its own
  `<dataset>_calibration.json`. See `Module_3_DRL_Environment_and_Agent.md` §5 for the definition
  actually in force.
- **`normalization_cycle_time`** — RF2's scaling constant `C` (Module 3 §5). Currently the literal
  `10000` (Meneghello et al.'s original value) for every dataset, not re-tuned per dataset —
  `_normalization_cycle_time_note` in the artifact records this explicitly.
- **`ppi_window_n`** — rolling-window size (in completed traces) for the live PPI `φ(t)` computed
  during simulation (Module 3 §5). Currently `20` for every dataset.

**Synthetic placeholder resource.** For every activity in `activities_at_risk`, a resource named
`"__NO_RESOURCE__"` is added to `resource_processing_times` with a neutral duration (that
activity's dataset-wide median) and, at simulation time, effectively infinite capacity — a
deadlock-avoidance mechanism, not a real staffing option. It is **excluded from
`resource_tier_map`**, so it is invisible to Module 3's cluster-layer observation, the agent's
action space, and Module 4's distillation — the agent can never select it and never sees it exist.

### How each artifact is consumed

**`resource_processing_times` → Module 2:**
When the agent assigns a specific resource to a specific activity, Module 2 uses that
resource's processing time for that activity to simulate the duration. Resources with no
entry for a given activity have never performed it and are ineligible for assignment.

**`resource_tier_map` → Module 3 (dual-layer observation system):**
At every decision step in the Gymnasium environment, two observations are maintained in
parallel:
- **Individual-layer observation** (given to the agent): per-resource availability,
  queue lengths, shock flag, time context.
- **Cluster-layer observation** (stored silently, never shown to the agent): for each
  (activity, tier) pair, the count of currently available resources in that tier — derived
  by aggregating individual availability through `resource_tier_map`.

After the agent assigns a specific resource to an activity, `resource_tier_map` immediately
translates this into a (activity, tier) action label. Both the cluster-layer observation and
the cluster-layer action are appended to the behavioral log that Module 4 consumes. The
agent is never aware of this parallel layer.

**`resource_tier_map` → Module 4:**
Module 4 receives a behavioral log of (cluster_observation, cluster_action) pairs generated
synchronously at runtime. It fits interpretable decision trees on this log to distill the
agent's policy into human-readable rules referencing tier availability and activity context
— not individual resource identities.

---

## 5. Validation / Sanity Checks (must be implemented as automated checks, not manual)

- Every resource that appears in the event log must have at least one valid
  (resource, activity) processing time entry, or be fully excluded with a logged reason.
- Every activity in the log must have at least one resource in at least one tier, **or** be
  listed in `activities_at_risk` and backed by the synthetic placeholder resource (Section 4) —
  not left with zero eligible resources and no fallback, which would deadlock the simulator.
  (An earlier version of this rule instead re-admitted the highest-support candidate that failed
  the `n_min` filter as a fallback; this silently let a 2-observation outlier become the sole,
  always-selected resource for an activity in one real case and was removed for that reason — see
  `RIMS_DRL_Modification_Plan.md` Step 40. The placeholder-resource path replaces it.)
- Report and log: number of tiers per activity, number of members per tier per activity,
  and any activities where the tier count was clamped below `n_tiers`.

---

## 6. Per-Dataset Notes

| Event log | Special handling |
|---|---|
| BPI12W | Filter to `W_`-prefixed activities before processing — automated `A_` and `O_` events do not represent human resource decisions. Only `COMPLETE` lifecycle events present — sojourn approximation applies. |
| BPI2017 | Filter to `W_`-prefixed activities. Has `start`/`complete` lifecycle pairs — true processing times available. Larger log — ensure the processing time computation is vectorized. |
