# Technical Reference — `empirical_preprocessing.py`

This document describes every function, every significant variable, and every non-trivial function call in the script. It is written for someone who already understands what the script does and wants to understand exactly how and why each piece works.

---

## Module-level declarations

### `TIER_LABELS`
```python
TIER_LABELS = {0: "slow", 1: "mid", 2: "fast", 3: "tier_3", 4: "tier_4"}
```
A lookup table mapping integer tier IDs to human-readable strings. Tier IDs are assigned by `compute_tiers` as integers starting from 0. This dict is used only when writing the `activity_tiers` summary block in the output JSON; the numeric ID is what all downstream code actually uses. Up to 5 tiers are named; anything beyond is auto-labelled `tier_N` inside `compute_tiers`.

### `_CSV_ALIASES`
```python
_CSV_ALIASES: dict[str, list[str]] = { ... }
```
A mapping from the four canonical column names the script uses internally (`case_id`, `activity`, `resource`, `timestamp`) to the list of raw column names they might appear under in real CSV files. The lookup is case-insensitive. `_normalise_csv_columns` iterates over this dict to build the rename map. Adding a new alias for a new dataset requires only appending to the relevant list here.

### `log`
```python
log = logging.getLogger(__name__)
```
The module-level logger. All output goes through this rather than `print` so severity can be filtered. The root logger is configured at startup with `logging.basicConfig(level=logging.INFO, ...)`, meaning INFO and above are shown. DEBUG lines would be suppressed unless the level is lowered.

### `PLACEHOLDER_RESOURCE`
```python
PLACEHOLDER_RESOURCE = "__NO_RESOURCE__"
```
The synthetic resource name injected into `resource_processing_times` (by `run`, see below) for every activity `validate` reports as having zero eligible resources after all filters. It is deliberately never added to `resource_tier_map`, so it never enters the cluster-layer observation Module 3 builds from that map — it exists purely so Modules 2/3's action space always has at least one valid `(resource, activity)` pair for that activity, instead of a permanently-empty one that would deadlock the simulator (Module 3's action mask is built directly from `resource_processing_times`'s keys).

### `_START_TS_CSV_ALIASES`
```python
_START_TS_CSV_ALIASES = ["start timestamp", "start:timestamp", "start time", "start_timestamp"]
```
Column-name aliases used only to detect an optional wide-format start timestamp in CSV logs (see the wide-format detection block below). Separate from `_CSV_ALIASES` because a start timestamp is optional — its absence must not raise, unlike a missing `case_id`/`activity`/`resource`/`timestamp`.

---

## `load_and_prepare`

**Signature:** `load_and_prepare(log_path: str, activity_prefix: str | None, activity_merge_map: dict[str, str] | None = None) -> tuple[pd.DataFrame, bool, bool]`

**Purpose:** Load the raw event log file (XES or CSV), normalise it into a uniform DataFrame, optionally merge activity names, detect whether true start→complete processing times are available (in either long or wide format), and optionally filter to a subset of activities.

**Returns:** `(df, has_true_pt_long, has_true_pt_wide)` where `df` has the columns `case_id`, `activity`, `resource`, `timestamp`, `lifecycle`, plus `start_timestamp` when a wide-format start timestamp was found; `has_true_pt_long` and `has_true_pt_wide` are Booleans that control which branch of Step 2 `run` selects (long-format true PT takes precedence over wide-format, which takes precedence over the sojourn approximation).

---

### Inside `load_and_prepare`

**`path = Path(log_path)`**  
Converts the string path to a `pathlib.Path` object so that `path.suffix` and `path.name` work cleanly without string manipulation.

**XES branch — `pm4py.read_xes(str(path))`**  
Reads the XES file into a pandas DataFrame using the PM4Py library. PM4Py preserves all XES attributes as columns, naming them with their XES namespace prefix (e.g., `case:concept:name`, `org:resource`, `start:timestamp`). The result already has a `time:timestamp` column of type `datetime64` (often with timezone info). The `str(path)` call is required because PM4Py does not accept `Path` objects directly.

**`rename` dict (XES branch)**  
Maps the PM4Py column names to the canonical names used throughout the rest of the script, including `"start:timestamp": "start_timestamp"` for wide-format logs (e.g. ConsultaDataMining201618, where the completion timestamp and the start timestamp of the same activity occurrence live on the same row). Applied via `df.rename(columns=rename)`. Only these named columns are renamed; all other PM4Py columns remain and are dropped later when the DataFrame is subset to its final columns.

**CSV branch — `pd.read_csv(str(path))`**  
Reads the raw CSV with no type coercion beyond what pandas defaults to. All columns are read as strings or numbers depending on their content. The result is immediately passed to `_normalise_csv_columns` to map column names to canonical form. A separate lookup then checks for an optional wide-format start-timestamp column via `_START_TS_CSV_ALIASES`, renaming it to `start_timestamp` if found (excluding whatever column was already claimed as `timestamp`, so the same column can't be both).

**Activity-name whitespace stripping — `df["activity"] = df["activity"].str.strip()`**  
Runs immediately after the format-specific load, before merging or any lifecycle/prefix logic. Some logs carry stray leading/trailing whitespace in `concept:name` that the corresponding mined Petri net's task names do not — BPI2017's `"W_Shortened completion "` (trailing space) is a concrete example. Left unstripped, the artifact's key for that activity would silently never match what the simulator asks about: `retrieve_possible_assignments()` in `gym_env.py` builds the action space directly from `resource_processing_times`'s keys, so a whitespace-mismatched key makes that activity's action space permanently empty — indistinguishable from a genuine zero-eligible-resource activity, except it isn't one, and there's no synthetic-placeholder safety net catching it because `validate()` (further downstream) checks against `all_activities`, which is captured from this same, now-stripped, `activity` column and would already agree with the mismatch — the discrepancy only becomes visible against an *external* reference (the Petri net) that this script never sees. This bug was found by running the actual simulator, not by anything internal to this script, which is why the fix has to live here rather than being self-evidently necessary from the script's own logic.

**Activity merging — `df["activity"] = df["activity"].replace(activity_merge_map)`**  
Applied only if `activity_merge_map` is given (e.g. `{"Traer informacion estudiante - banner": "Submit Request", "Radicar Solicitud Homologacion": "Submit Request"}` for ConsultaDataMining201618). Runs after whitespace stripping so merge-map keys don't need to anticipate stray whitespace, and before lifecycle/prefix detection so every downstream computation sees only the merged name. `pd.Series.replace` with a dict replaces exact-match values; activities not present as keys are left untouched. This is generic, reusable infrastructure — not Consulta-specific — for collapsing two activity names that should be treated as one.

**`lifecycle_col = next(..., None)`**  
Searches `df.columns` for any column name that contains the substring `"lifecycle"` (case-insensitive), excluding `"case:lifecycle:transition"` which is a PM4Py case-level attribute, not an event-level lifecycle marker. Returns the first match or `None`. The `next(..., None)` pattern is Python's idiomatic way of extracting the first element of a generator with a fallback.

**`lc_lower = df["lifecycle"].dropna().str.lower()`**  
Creates a lowercase Series of the lifecycle column values, dropping nulls first so that `.str.lower()` does not produce NaN. This Series is used twice: once to build the sets of start/complete activities for the `has_true_pt_long` check, and implicitly in downstream functions via the same string comparison.

**`has_true_pt_long` detection**  
```python
start_acts = set(df[lc_lower == "start"]["activity"].dropna().unique())
complete_acts = set(df[lc_lower == "complete"]["activity"].dropna().unique())
has_true_pt_long = len(start_acts & complete_acts) > 0
```
`has_true_pt_long` is `True` only if there exists at least one activity name that appears in both a `"start"` row and a `"complete"` row (the two-row, "long" format). This is stricter than just checking whether both values appear anywhere in the column. The stricter check prevents false positives on logs like the helpdesk CSV, where the value `"Start"` appears as the lifecycle field of an artificial trace-start activity row, not as a genuine start event for any work activity. `start_acts & complete_acts` is the Python set intersection operator; its length is zero when no activity appears in both sets.

**Wide-format detection**  
```python
has_start_ts_col = "start_timestamp" in df.columns
if has_start_ts_col:
    if pd.api.types.is_object_dtype(df["start_timestamp"]):
        df["start_timestamp"] = pd.to_datetime(df["start_timestamp"], utc=False, errors="coerce")
    if hasattr(df["start_timestamp"].dtype, "tz") and df["start_timestamp"].dt.tz is not None:
        df["start_timestamp"] = df["start_timestamp"].dt.tz_localize(None)
has_true_pt_wide = has_start_ts_col and df["start_timestamp"].notna().any()
```
Parallel to the long-format check, but for logs where the start and completion timestamps of the same activity occurrence live on the *same row* (e.g. ConsultaDataMining201618's `start:timestamp` alongside `time:timestamp`, with `lifecycle:transition` uniformly `"complete"` — the long-format check above would correctly report `has_true_pt_long = False` for such a log, since there's no `"start"` lifecycle value at all). `has_true_pt_wide` is `True` only if a `start_timestamp` column exists *and* has at least one non-null value after parsing. The same tz-stripping treatment as the main `timestamp` column is applied here (`errors="coerce"` additionally turns any unparseable value into `NaT` rather than raising, since this column is optional and best-effort).

**`df[df["activity"].str.startswith(activity_prefix)]`**  
Boolean mask using pandas `.str.startswith()`. For BPI12W and BPI2017, the prefix is `"W_"`, which selects only human-work activities and discards automated `A_` and `O_` events. When `activity_prefix` is `None`, this block is skipped entirely.

**`df.dropna(subset=["resource"])`**  
Removes rows where the `resource` column is null. System-generated events (e.g., automated lifecycle transitions, the artificial "Start" row in the helpdesk CSV) often have no resource recorded. Keeping them would produce NaN group keys in the processing-time groupbys downstream.

**Timestamp normalisation**  
```python
if pd.api.types.is_object_dtype(df["timestamp"]):
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
if hasattr(df["timestamp"].dtype, "tz") and df["timestamp"].dt.tz is not None:
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
```
Two separate guards. The first converts string timestamps (from CSV) to `datetime64`. The second strips timezone information from already-typed timestamps (common in XES files, which embed timezone in the ISO 8601 string). Stripping the timezone is necessary because pandas cannot subtract a tz-aware timestamp from a tz-naive one, which would otherwise crash Step 2.

**Column selection**  
```python
keep_cols = ["case_id", "activity", "resource", "timestamp", "lifecycle"]
if has_start_ts_col:
    keep_cols.append("start_timestamp")
df = df[keep_cols].copy()
```
Explicitly selects and reorders the columns, discarding all others — `start_timestamp` is kept only when it was actually found, so downstream functions can rely on `"start_timestamp" in df.columns` to mean "wide-format data is present" without a separate flag. The `.copy()` call creates a fresh DataFrame rather than a view, which prevents pandas `SettingWithCopyWarning` when downstream functions add columns to it.

**`df.sort_values(["case_id", "timestamp"]).reset_index(drop=True)`**  
Sorts the entire DataFrame by case then by time. This ordering is a precondition for the sojourn-time approximation in `_compute_sojourn_pt`, which uses `.shift(1)` to look at the previous row within each case. Without this sort, `.shift(1)` would compare events across different cases or in wrong order. `reset_index(drop=True)` assigns a clean 0-based integer index so that row references in downstream code are unambiguous.

---

## `_normalise_csv_columns`

**Signature:** `_normalise_csv_columns(df: pd.DataFrame) -> pd.DataFrame`

**Purpose:** Map whatever column names appear in a CSV to the four canonical names (`case_id`, `activity`, `resource`, `timestamp`) using the `_CSV_ALIASES` lookup. Raises a `ValueError` with a clear message if a required column cannot be matched.

---

### Inside `_normalise_csv_columns`

**`lower_cols = {c.lower().strip(): c for c in df.columns}`**  
A reverse lookup: maps each lowercased, whitespace-stripped column name back to the original column name as it appears in the DataFrame. This allows the alias lookup to be case-insensitive without altering the actual column names before the rename.

**`matched = next((lower_cols[a] for a in aliases if a in lower_cols), None)`**  
Iterates over the alias list in order, returns the original column name for the first alias that matches a key in `lower_cols`, or `None` if none match. The ordering within each alias list in `_CSV_ALIASES` is therefore a priority ordering: earlier aliases are preferred.

**`missing` list**  
Collects canonical column names for which no alias matched. A second pass (`still_missing`) re-checks whether the canonical name itself was already present in `df.columns` (which the earlier `if canonical in df.columns: continue` guard would have handled), then raises if anything is truly unresolvable.

---

## `_compute_true_pt`

**Signature:** `_compute_true_pt(df: pd.DataFrame) -> tuple[pd.DataFrame, list]`

**Purpose:** For logs where the same case-activity-resource combination has both a `start` and a `complete` event, compute the elapsed time between each matched pair. Returns a flat records DataFrame with columns `resource`, `activity`, `pt` (processing time in seconds), plus a list of excluded event pairs.

---

### Inside `_compute_true_pt`

**`starts` and `completes`**  
Two separate DataFrames filtered to `lifecycle == "start"` and `lifecycle == "complete"` respectively, keeping only the four columns needed for pairing. The `.copy()` prevents pandas from raising warnings when new columns are added.

**`key = ["case_id", "activity", "resource"]`**  
The three-column grouping key for pairing. Two events are candidates for a match only if they share the same case, the same activity name, and the same resource. This rules out, for example, two different resources working on the same activity in the same case, or the same resource working the same activity in two different cases.

**`starts["_pair"] = starts.groupby(key).cumcount()`**  
`groupby(key).cumcount()` assigns a sequential integer (0, 1, 2, …) to each row within each group, in the order the DataFrame is sorted. Because both `starts` and `completes` are sorted by `key + ["timestamp"]` just before this line, the `_pair` integer represents the chronological rank of each event within its group. The first `start` event in a group gets `_pair = 0`, the second gets `_pair = 1`, and so on. The same numbering is applied independently to the completes DataFrame.

**`starts.merge(completes, on=key + ["_pair"], suffixes=("_s", "_c"))`**  
Inner join matching start and complete rows that share the same group key and the same `_pair` integer. This implements a "first start matches first complete" pairing within each group. It is an inner join, so any unmatched starts (no corresponding complete) or unmatched completes (no corresponding start) are silently dropped. `suffixes=("_s", "_c")` disambiguates the two `timestamp` columns, producing `timestamp_s` (start) and `timestamp_c` (complete).

**`merged["pt"] = (merged["timestamp_c"] - merged["timestamp_s"]).dt.total_seconds()`**  
Subtracts two `datetime64` columns, producing a `timedelta64` Series, then calls `.dt.total_seconds()` to convert to a float number of seconds. This is the raw processing time for each matched pair.

**`bad = merged[merged["pt"] <= 0]`**  
Identifies pairs where the complete timestamp is not strictly after the start timestamp. This can happen due to clock skew, data entry errors, or same-second timestamps. These are collected into `excluded` and the corresponding rows are dropped from `records`.

---

## `_compute_true_pt_wide`

**Signature:** `_compute_true_pt_wide(df: pd.DataFrame) -> tuple[pd.DataFrame, list]`

**Purpose:** For logs where the start and completion timestamps of the same activity occurrence live on the same row (`start_timestamp` alongside `timestamp`), compute the elapsed time directly per row — no cross-row pairing is needed, unlike `_compute_true_pt`. Returns a flat records DataFrame with columns `resource`, `activity`, `pt`, plus a list of excluded rows.

---

### Inside `_compute_true_pt_wide`

**`rows = df.dropna(subset=["start_timestamp"]).copy()`**  
Drops any row without a parsed `start_timestamp` (these would have come from unparseable values coerced to `NaT` in `load_and_prepare`, or from a log where only some rows carry a start timestamp). The `.copy()` avoids a `SettingWithCopyWarning` on the next line.

**`rows["pt"] = (rows["timestamp"] - rows["start_timestamp"]).dt.total_seconds()`**  
Same subtraction-then-`.dt.total_seconds()` pattern as `_compute_true_pt`, but row-local: no `groupby`/`cumcount`/`merge` machinery is needed at all, since both timestamps are already on the same row by construction. This is structurally simpler than the long-format pairing, at the cost of only applying to logs that actually carry a same-row start timestamp.

**`bad = rows[rows["pt"] <= 0]`**  
Same non-positive-duration exclusion as `_compute_true_pt`, tagged with the distinct reason `"zero_or_negative_true_pt_wide"` so the artifact's `excluded_pairs` list can distinguish which detection path produced the exclusion.

---

## `_compute_sojourn_pt`

**Signature:** `_compute_sojourn_pt(df: pd.DataFrame) -> tuple[pd.DataFrame, list]`

**Purpose:** For logs where only a single timestamp per event exists (no start/complete pair), approximate the processing time of an event as the elapsed time since the previous event in the same case. This is a sojourn time: it includes both waiting time and actual working time. The limitation is documented in the output artifact's `config_used.processing_time_mode` field.

---

### Inside `_compute_sojourn_pt`

**`comp = df[df["lifecycle"].str.lower() == "complete"].copy()`**  
Filters to complete events. In pure-complete logs (e.g., helpdesk CSV where all work events have lifecycle `"complete"`), this selects essentially all rows except the artificial start marker. In logs with mixed lifecycles, it discards the start events, which is correct because the sojourn approximation only makes sense when measuring time between consecutive recorded events.

**`comp["prev_ts"] = comp.groupby("case_id")["timestamp"].shift(1)`**  
Within each case, shifts the `timestamp` column down by one position. The result, `prev_ts`, holds the timestamp of the immediately preceding event in the same case. For the first event in each case, `prev_ts` is `NaN` because there is no predecessor. `groupby("case_id")` ensures the shift boundary respects case boundaries: the first event of case 2 does not accidentally use the last event of case 1 as its predecessor.

**`comp["pt"] = (comp["timestamp"] - comp["prev_ts"]).dt.total_seconds()`**  
Computes the elapsed time in seconds between each event and its predecessor. `NaN - anything` produces `NaN`, so first-events-in-case produce `NaN` here. These are removed in the next step.

**`records = comp.dropna(subset=["prev_ts"])`**  
Removes the first event in each case (where `prev_ts` is `NaN`). This is necessary rather than arbitrary: the first event has no predecessor to measure from, so no sojourn time can be computed for it.

**`records = records[records["pt"] > 0]`**  
Drops any remaining rows with zero or negative elapsed time. These can arise from duplicate timestamps or from two events that were logged simultaneously (common in batch-entry systems).

---

## `aggregate`

**Signature:** `aggregate(records: pd.DataFrame, n_min: int, cap_pct: float = 95.0) -> tuple[dict, dict, list]`

**Purpose:** For each `(resource, activity)` pair in the records DataFrame, apply the three-stage aggregation pipeline: outlier capping → median computation → minimum-support filter. Returns the two output maps and the exclusion log.

---

### Inside `aggregate`

**`resource_pt: dict[str, dict[str, float]]`**  
The primary output: a nested dict keyed by resource name, then activity name, with the capped-median processing time in seconds as the value. This is the `resource_processing_times` field in the JSON artifact and the dict consumed by Module 2 for duration sampling.

**`support_counts: dict[str, dict[str, int]]`**  
Parallel structure to `resource_pt`, storing the raw number of observations (before capping) for each `(resource, activity)` pair that passed all filters. Used to populate the `support_counts` field in the artifact, which the thesis reports as evidence of data coverage.

**`excluded_by_activity: dict[str, list[dict]]`**  
A temporary structure that accumulates, for each activity, all the `(resource, n, median_pt)` candidates that were rejected by the n_min or median filter. This is only needed by the fallback logic and is not written to the artifact directly.

**`records.groupby(["resource", "activity"])`**  
Groups all raw processing-time records by the two-key combination. For each group, the loop body runs once. The group key `(resource, activity)` identifies the pair being evaluated.

**`pts = grp["pt"].values`**  
Extracts the numpy array of raw processing times for this group. Using `.values` rather than keeping it as a Series is a minor optimisation: subsequent numpy calls (`np.percentile`, `np.minimum`, `np.median`) operate slightly faster on plain arrays than on Series objects.

**`cap = float(np.percentile(pts, cap_pct))`**  
Computes the `cap_pct`-th percentile (default 95th) of the raw processing times for this pair. All values above this threshold are clipped to this value in the next step. Computing it per-pair rather than globally means each pair's cap reflects its own distribution, not a global one.

**`pts_capped = np.minimum(pts, cap)`**  
Element-wise minimum between each raw value and the cap. Values at or below the cap are unchanged; values above the cap are replaced by the cap value. This is outlier capping without removal: every observation still contributes to the median, but extreme outliers no longer dominate it.

**`median_pt = float(np.median(pts_capped))`**  
The median of the capped values. The median is preferred over the mean because it is robust to the remaining variation in the capped distribution. For skewed processing-time distributions (common in real logs), the median better represents a "typical" duration.

**Fallback block**  
```python
admitted_by_activity = {
    activity
    for resource, acts in resource_pt.items()
    for activity in acts
}
for activity, candidates in excluded_by_activity.items():
    if activity not in admitted_by_activity:
        ...
```
`admitted_by_activity` is a flat set of all activity names that have at least one resource in `resource_pt` after the main loop. For any activity in `excluded_by_activity` that is not in this set (meaning every resource for that activity was filtered out), the fallback selects the candidate with the highest `n` (most observations) among those with a positive `median_pt` and re-admits them. The re-admission re-runs the capping and median computation from scratch on the original records to ensure the stored value is computed consistently. The corresponding entry in `excluded` is annotated with `"_FALLBACK_READMITTED"` so the artifact records that this pair was re-admitted despite failing the filter.

---

## `compute_tiers`

**Signature:** `compute_tiers(resource_pt: dict, n_tiers: int) -> tuple[dict, dict]`

**Purpose:** For each activity independently, assign each of its eligible resources to one of `n_tiers` performance tiers using rank-based quantile binning. Tier 0 is the slowest group; tier `n_tiers - 1` is the fastest group.

---

### Inside `compute_tiers`

**`activity_res: dict[str, dict[str, float]]`**  
An inverted view of `resource_pt`: instead of `resource → activity → pt`, this is `activity → resource → pt`. The inversion is needed because tiering is done per-activity: for each activity, all resources that have performed it are ranked and split into groups.

**`effective = min(n_tiers, n_eligible)`**  
The actual number of tiers used for this activity. If an activity has fewer eligible resources than `n_tiers`, the number of tiers is clamped down. For example, `W_Beoordelen fraude` in BPI12W has only 4 eligible resources with `n_tiers=3`, which is fine; `Create SW anomaly` in the helpdesk log has only 2, so it is clamped to 2 tiers.

**`sorted_res = sorted(res_pt_map.items(), key=lambda x: x[1])`**  
Sorts `(resource, median_pt)` pairs in ascending order of median processing time. After sorting, index 0 holds the fastest resource (lowest pt) and index `n-1` holds the slowest. The sort direction matters: the tier formula is written assuming this ascending order.

**`rank_fast` and `rank_slow`**  
`rank_fast` is the loop index, running from 0 (fastest resource) to `n-1` (slowest). `rank_slow = n - 1 - rank_fast` is its mirror: 0 for the slowest, `n-1` for the fastest. The tier assignment uses `rank_slow`:

```python
tier = min(int(rank_slow * effective / n), effective - 1)
```

This maps `rank_slow = 0` (slowest resource) to `tier = 0` ("slow"), and `rank_slow = n-1` (fastest resource) to `tier ≈ effective - 1` ("fast"). The formula is a linear mapping from rank space `[0, n)` to tier space `[0, effective)`. The `min(..., effective - 1)` clamp prevents floating-point rounding from producing `tier = effective` for the highest rank.

**`tier_groups: dict[int, list[str]]`**  
Groups resources by their assigned tier ID for the purpose of computing the per-tier summary statistics. Built after all individual tier assignments are done.

**`mean_pt = float(np.mean([res_pt_map[r] for r in members]))`**  
The mean of the median processing times of all resources in a tier. This is a mean-of-medians: each `res_pt_map[r]` is already a capped median from `aggregate`, so this is an average over those representative values.

**`resource_tier_map.setdefault(resource, {})[activity] = tier`**  
Populates the second output dict. `.setdefault(resource, {})` returns the existing inner dict for this resource if it exists, or inserts and returns an empty dict if not. This avoids the need for an explicit `if resource not in resource_tier_map` check.

---

## `validate`

**Signature:** `validate(resource_pt: dict, activity_tiers: dict, all_activities: set) -> list[str]`

**Purpose:** Check that every activity observed in the raw log has at least one eligible resource in `resource_pt`. Activities that do not are collected into the `at_risk` list, logged as warnings, and returned so they can be embedded in the artifact under `activities_at_risk`. This function does not raise an exception; it treats the at-risk list as a documented constraint rather than a fatal error.

---

### Inside `validate`

**`all_activities`**  
The complete set of unique activity names seen in the filtered, prepared DataFrame — computed in `run` before any processing-time computation. This is the reference set: every activity in it should have at least one resource after all the filtering.

**`any(activity in acts for acts in resource_pt.values())`**  
For each activity name from `all_activities`, checks whether it appears as a key in any resource's inner dict in `resource_pt`. This is an existence check, not a value check. `resource_pt.values()` yields the inner dicts; `activity in acts` tests membership in that dict's keys.

---

## `run`

**Signature:**
```python
run(
    log_path: str,
    output_path: str,
    activity_prefix: str | None = None,
    n_tiers: int = 3,
    n_min: int = 3,
    cap_pct: float = 95.0,
    activity_merge_map: dict[str, str] | None = None,
    exclude_activities: list[str] | None = None,
    p_star: float | None = None,
    normalization_cycle_time: float = 10000.0,
    ppi_window_n: int = 20,
) -> dict
```

**Purpose:** Orchestrates the full pipeline (load → merge/strip → true-PT-long / true-PT-wide /
sojourn → aggregate → optional exclusion → tier → validate → placeholder-resource injection)
and writes the result to a JSON file consumed by Modules 2/3. This is the only function an
external caller needs to invoke. It can also be called from another Python module (e.g., a
notebook) with the return value used directly, without touching the filesystem.

---

### Inside `run`

**`load_and_prepare(log_path, activity_prefix, activity_merge_map)`**  
Now returns three values, `(df, has_true_pt_long, has_true_pt_wide)` — see that function's
own section above. `activity_merge_map` is threaded straight through from `run`'s own
parameter.

**`all_activities = set(df["activity"].unique())`**  
Captured immediately after `load_and_prepare` so it reflects the activity universe *after*
whitespace stripping and merging but before any processing-time records are computed or
filtered. It is the reference set `validate` checks at the end, so it must already reflect
the merged/stripped names — otherwise a merged activity like Consulta's `"Submit Request"`
would never appear in `all_activities` and `validate` couldn't report it as at-risk.

**Mode selection**  
```python
if has_true_pt_long:
    mode = "true"
    records, excluded_stage1 = _compute_true_pt(df)
elif has_true_pt_wide:
    mode = "true_wide"
    records, excluded_stage1 = _compute_true_pt_wide(df)
else:
    mode = "sojourn_approximation"
    records, excluded_stage1 = _compute_sojourn_pt(df)
```
Precedence is long-format true PT, then wide-format true PT, then the sojourn
approximation — the first one that the log's structure actually supports. `mode` is recorded
verbatim in `config_used.processing_time_mode` in the artifact, so it's always visible which
path a given dataset took.

**`records`**  
A flat DataFrame with three columns: `resource`, `activity`, `pt`. It is the intermediate
output of whichever of the three functions above ran, containing one row per valid, positive
processing-time observation. This DataFrame is consumed by `aggregate` and then discarded.

**`excluded_stage1` and `excluded_stage2`**  
Two lists of exclusion dicts, one from the raw observation level (zero/negative individual
measurements) and one from the aggregation level (low support or nonpositive median after
capping). They are concatenated into the `excluded_pairs` field of the artifact so the full
exclusion history is available in one place.

**Explicit activity exclusion**  
```python
if exclude_activities:
    for resource in list(resource_pt.keys()):
        for act in exclude_activities:
            resource_pt[resource].pop(act, None)
            support_counts.get(resource, {}).pop(act, None)
        if not resource_pt[resource]:
            del resource_pt[resource]
```
Runs after `aggregate()`, before `compute_tiers()`. For each activity named in
`exclude_activities` (e.g. Consulta's merged `"Submit Request"`, whose `resource` values are
students' own account IDs, not real staff), strips it out of every resource's inner dict in
`resource_pt` (and the parallel `support_counts`), then drops any resource left with an empty
inner dict. Because this runs *before* `compute_tiers`, the excluded activity is guaranteed
to have zero eligible resources by the time `validate` checks it — it will show up in
`activities_at_risk` and get the placeholder-resource treatment below, rather than needing a
separate code path.

**`compute_tiers(resource_pt, n_tiers)`**  
Unchanged internally, but now runs on `resource_pt` *after* the exclusion step above, so an
excluded activity never contributes tier data.

**`at_risk = validate(resource_pt, activity_tiers, all_activities)`**  
Unchanged internally. With the exclusion step above, this list now also picks up any
explicitly-excluded activity, in addition to activities that were always going to be at-risk
from `aggregate`'s own n_min/median filtering.

**Synthetic placeholder-resource injection**  
```python
if at_risk:
    all_pts = [pt for acts in resource_pt.values() for pt in acts.values()]
    placeholder_pt = float(np.median(all_pts)) if all_pts else 3600.0
    for activity in at_risk:
        resource_pt.setdefault(PLACEHOLDER_RESOURCE, {})[activity] = placeholder_pt
        support_counts.setdefault(PLACEHOLDER_RESOURCE, {})[activity] = 0
```
Runs **after** `validate()` (and therefore after `compute_tiers()`) — this ordering is what
guarantees `PLACEHOLDER_RESOURCE` is never part of `resource_tier_map`: `compute_tiers` has
already finished building that structure from the pre-injection `resource_pt` by the time this
block mutates it further. `placeholder_pt` is the dataset-wide median processing time across
every already-eligible `(resource, activity)` pair (a neutral, non-privileged duration — not
tied to any specific activity) — falling back to a flat `3600.0` seconds only in the
degenerate case where `resource_pt` is entirely empty. `support_counts[PLACEHOLDER_RESOURCE]`
is set to `0` for every activity it covers, honestly reflecting that it represents zero real
observations. Because `resource_pt[PLACEHOLDER_RESOURCE]` only ever gets entries for
activities literally listed in `at_risk`, this resource can never appear as eligible for any
activity that already has real resources — `retrieve_possible_assignments()` in `gym_env.py`
builds the action space purely from these `(resource, activity)` key pairs, so there is no
separate scoping mechanism needed beyond this dict's own keys.

**The artifact dict**  
The final JSON structure. Ten top-level keys map to the outputs of the pipeline stages
(four more than before this round of changes):
- `activity_tiers` — from `compute_tiers`, summary view indexed by activity then tier ID
- `resource_tier_map` — from `compute_tiers`, lookup indexed by resource then activity (never
  contains `PLACEHOLDER_RESOURCE`)
- `resource_processing_times` — from `aggregate` plus the placeholder-injection step, the
  primary Module 2/3 input (*does* contain `PLACEHOLDER_RESOURCE` when `activities_at_risk`
  is non-empty)
- `support_counts` — from `aggregate` plus the placeholder-injection step, evidence of data
  coverage
- `excluded_pairs` — stage 1 + stage 2 exclusions concatenated
- `activities_at_risk` — from `validate`, activities with zero eligible resources (before the
  placeholder-injection step neutralises the practical consequence of that)
- **`P_star`** — passed straight through from the `p_star` parameter; **not** computed by
  this script (see `compute_p_star.py` for that) — this is purely a place for the
  already-computed value to live alongside everything else Module 3 needs
- **`normalization_cycle_time`** — passed straight through from the parameter; the `C`
  scaling constant used by Module 3's inverse-cycle-time reward term
- **`_normalization_cycle_time_note`** — a fixed human-readable string flagging that the
  value is Meneghello et al.'s original default, not yet re-tuned per dataset
- **`ppi_window_n`** — passed straight through from the parameter; the rolling-window size (in
  completed traces) Module 3 uses to compute its live performance indicator `phi(t)`

**`config_used`**  
Now also records `activity_merge_map` and `exclude_activities` (whatever was passed in,
possibly `None`), alongside the pre-existing fields — so a regenerated artifact always shows
exactly which merge/exclusion configuration produced it.

**`out.parent.mkdir(parents=True, exist_ok=True)`**  
Creates the output directory and any missing parent directories if they do not exist. `exist_ok=True` means no error is raised if the directory already exists. Without this, writing to `artifacts/bpi12w.json` when `artifacts/` does not yet exist would fail.

**`json.dump(artifact, f, indent=2)`**  
Writes the artifact as pretty-printed JSON with 2-space indentation. This makes the file human-readable and diff-friendly in version control. The `indent` argument is the only formatting choice; no custom encoder is needed because all values in the artifact are standard Python types (str, int, float, list, dict).

---

## `__main__` block

Runs only when the script is invoked directly (`python empirical_preprocessing.py ...`). Uses `argparse` to parse the CLI arguments.

**`argparse.ArgumentDefaultsHelpFormatter`**  
A built-in formatter class that automatically appends ` (default: X)` to each help string when `--help` is run. No extra code needed; the defaults shown are whatever is passed as `default=` in each `add_argument` call.

**`args.activity_prefix`** maps to `--activity-prefix` (hyphen in CLI, underscore in Python attribute, this is standard argparse behaviour).

**`--merge-activities OLD=NEW ...` / `activity_merge_map`**  
```python
activity_merge_map = None
if args.merge_activities:
    activity_merge_map = dict(pair.split("=", 1) for pair in args.merge_activities)
```
Accepts one or more `OLD=NEW` strings (`nargs="+"`) and turns them into the dict `run()`
expects. `split("=", 1)` limits the split to the first `=`, so an activity name that itself
contains `=` would still parse correctly (the new name cannot contain `=`, but no real
activity name in the datasets used here does).

**`--exclude-activities ACT ...`**  
Passed straight through as a list (`nargs="+"`) to `run()`'s `exclude_activities` parameter.

**`--p-star`, `--normalization-cycle-time`, `--ppi-window-n`**  
Straightforward `type=float`/`type=int` arguments passed straight through to the
correspondingly-named `run()` parameters, with the same defaults (`None`, `10000.0`, `20`).

**`sys.exit(1)`**  
Exits with a non-zero status code when an exception propagates out of `run`. This signals failure to any calling shell script or CI system. The exception message is logged via `log.error` before exit so the cause is visible.
