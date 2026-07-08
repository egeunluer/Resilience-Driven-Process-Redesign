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

---

## `load_and_prepare`

**Signature:** `load_and_prepare(log_path: str, activity_prefix: str | None) -> tuple[pd.DataFrame, bool]`

**Purpose:** Load the raw event log file (XES or CSV), normalise it into a uniform five-column DataFrame, detect whether true start→complete processing times are available, and optionally filter to a subset of activities.

**Returns:** `(df, has_true_pt)` where `df` always has exactly the columns `case_id`, `activity`, `resource`, `timestamp`, `lifecycle`; and `has_true_pt` is a Boolean that controls which branch of Step 2 runs.

---

### Inside `load_and_prepare`

**`path = Path(log_path)`**  
Converts the string path to a `pathlib.Path` object so that `path.suffix` and `path.name` work cleanly without string manipulation.

**XES branch — `pm4py.read_xes(str(path))`**  
Reads the XES file into a pandas DataFrame using the PM4Py library. PM4Py preserves all XES attributes as columns, naming them with their XES namespace prefix (e.g., `case:concept:name`, `org:resource`). The result already has a `time:timestamp` column of type `datetime64` (often with timezone info). The `str(path)` call is required because PM4Py does not accept `Path` objects directly.

**`rename` dict (XES branch)**  
Maps the four PM4Py column names to the canonical names used throughout the rest of the script. Applied via `df.rename(columns=rename)`. Only these four columns are renamed; all other PM4Py columns remain and are dropped at the end of the function when the DataFrame is subset to five columns.

**CSV branch — `pd.read_csv(str(path))`**  
Reads the raw CSV with no type coercion beyond what pandas defaults to. All columns are read as strings or numbers depending on their content. The result is immediately passed to `_normalise_csv_columns` to map column names to canonical form.

**`lifecycle_col = next(..., None)`**  
Searches `df.columns` for any column name that contains the substring `"lifecycle"` (case-insensitive), excluding `"case:lifecycle:transition"` which is a PM4Py case-level attribute, not an event-level lifecycle marker. Returns the first match or `None`. The `next(..., None)` pattern is Python's idiomatic way of extracting the first element of a generator with a fallback.

**`lc_lower = df["lifecycle"].dropna().str.lower()`**  
Creates a lowercase Series of the lifecycle column values, dropping nulls first so that `.str.lower()` does not produce NaN. This Series is used twice: once to build the sets of start/complete activities for the `has_true_pt` check, and implicitly in downstream functions via the same string comparison.

**`has_true_pt` detection**  
```python
start_acts = set(df[lc_lower == "start"]["activity"].dropna().unique())
complete_acts = set(df[lc_lower == "complete"]["activity"].dropna().unique())
has_true_pt = len(start_acts & complete_acts) > 0
```
`has_true_pt` is `True` only if there exists at least one activity name that appears in both a `"start"` row and a `"complete"` row. This is stricter than just checking whether both values appear anywhere in the column. The stricter check prevents false positives on logs like the helpdesk CSV, where the value `"Start"` appears as the lifecycle field of an artificial trace-start activity row, not as a genuine start event for any work activity. `start_acts & complete_acts` is the Python set intersection operator; its length is zero when no activity appears in both sets.

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

**`df[["case_id", "activity", "resource", "timestamp", "lifecycle"]].copy()`**  
Explicitly selects and reorders the five columns, discarding all others. The `.copy()` call creates a fresh DataFrame rather than a view, which prevents pandas `SettingWithCopyWarning` when downstream functions add columns to it.

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

**Signature:** `run(log_path, output_path, activity_prefix, n_tiers, n_min, cap_pct) -> dict`

**Purpose:** Orchestrates the full five-stage pipeline and writes the result to a JSON file. This is the only function an external caller needs to invoke. It can also be called from another Python module (e.g., a notebook) with the return value used directly, without touching the filesystem.

---

### Inside `run`

**`all_activities = set(df["activity"].unique())`**  
Captured immediately after `load_and_prepare` so it reflects the activity universe before any processing-time records are computed or filtered. It is used only by `validate` at the end, to check whether any activity was lost during processing.

**`records`**  
A flat DataFrame with three columns: `resource`, `activity`, `pt`. It is the intermediate output of either `_compute_true_pt` or `_compute_sojourn_pt`, containing one row per valid, positive processing-time observation. This DataFrame is consumed by `aggregate` and then discarded.

**`excluded_stage1` and `excluded_stage2`**  
Two lists of exclusion dicts, one from the raw observation level (zero/negative individual measurements) and one from the aggregation level (low support or nonpositive median after capping). They are concatenated into the `excluded_pairs` field of the artifact so the full exclusion history is available in one place.

**The artifact dict**  
The final JSON structure. All six top-level keys map directly to the outputs of the pipeline stages:
- `activity_tiers` — from `compute_tiers`, summary view indexed by activity then tier ID
- `resource_tier_map` — from `compute_tiers`, lookup indexed by resource then activity
- `resource_processing_times` — from `aggregate`, the primary Module 2 input
- `support_counts` — from `aggregate`, evidence of data coverage
- `excluded_pairs` — stage 1 + stage 2 exclusions concatenated
- `activities_at_risk` — from `validate`, activities with zero eligible resources

**`out.parent.mkdir(parents=True, exist_ok=True)`**  
Creates the output directory and any missing parent directories if they do not exist. `exist_ok=True` means no error is raised if the directory already exists. Without this, writing to `artifacts/bpi12w.json` when `artifacts/` does not yet exist would fail.

**`json.dump(artifact, f, indent=2)`**  
Writes the artifact as pretty-printed JSON with 2-space indentation. This makes the file human-readable and diff-friendly in version control. The `indent` argument is the only formatting choice; no custom encoder is needed because all values in the artifact are standard Python types (str, int, float, list, dict).

---

## `__main__` block

Runs only when the script is invoked directly (`python empirical_preprocessing.py ...`). Uses `argparse` to parse four CLI arguments.

**`argparse.ArgumentDefaultsHelpFormatter`**  
A built-in formatter class that automatically appends ` (default: X)` to each help string when `--help` is run. No extra code needed; the defaults shown are whatever is passed as `default=` in each `add_argument` call.

**`args.activity_prefix`** maps to `--activity-prefix` (hyphen in CLI, underscore in Python attribute, this is standard argparse behaviour).

**`sys.exit(1)`**  
Exits with a non-zero status code when an exception propagates out of `run`. This signals failure to any calling shell script or CI system. The exception message is logged via `log.error` before exit so the cause is visible.
