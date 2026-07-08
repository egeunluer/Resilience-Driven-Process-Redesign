# Technical Reference — `collect_behavioral_log.py`

This document describes every function, every significant variable, and every non-trivial function call in the script. It is written for someone who already understands what the script does and wants to understand exactly how and why each piece works.

---

## Purpose and position in the pipeline

This script runs **after** DRL training is complete. It loads the frozen trained model, runs it through a controlled set of simulation episodes, and writes a flat CSV — the **behavioral log** — that Module 4 (policy distillation) consumes as its sole input.

It is intentionally separate from the training loop. During training the policy is updated at every step; the behavioral log must reflect the *final* policy, not an average across all training checkpoints. Using a frozen model with `deterministic=True` prediction gives a clean, reproducible dataset.

---

## Module-level docstring

The module docstring (lines 1–30) documents the script's purpose, explains why collection is post-training, describes the output CSV schema (columns and their meaning), and shows a complete example CLI call. It is the first thing a reader should read before looking at any function.

---

## `parse_args()`

**Signature:** `parse_args() -> argparse.Namespace`

**Purpose:** Parse command-line arguments and return them as a namespace object.

**Arguments defined:**

| Argument | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `--name-log` | str | yes | — | Dataset name; must match the subfolder name under `example/` |
| `--n-traces` | str | no | `'from_input_data'` | Number of traces per episode, or the literal string `'from_input_data'` to read it from the input JSON |
| `--artifact` | str | yes | — | Path to the Module 1 JSON artifact (`resource_processing_times`, `resource_tier_map`) |
| `--model` | str | yes | — | Path to the saved `MaskablePPO` model file, without the `.zip` extension that SB3 appends automatically |
| `--output` | str | yes | — | Full path for the output CSV file |
| `--n-episodes` | int | no | 50 | Number of collection episodes to run |
| `--shock` | flag | no | False | If present, shocks are injected during collection using `build_shock_config()` |
| `--threshold` | int | no | 0 | Passed to `gym_env` unchanged; used only for datasets that have multiple threshold variants |

`argparse.ArgumentParser` is Python's standard library argument parser. `add_argument` with `action="store_true"` means `--shock` sets the attribute to `True` when present and `False` when absent, with no value required on the command line.

---

## `build_shock_config()`

**Signature:** `build_shock_config() -> ShockConfig`

**Purpose:** Construct and return the `ShockConfig` used during collection episodes.

**Why `mode='eval'`:** In eval mode, `shock_scheduler` fires exactly one shock per episode at a fixed `eval_onset_time`, holds it for `eval_duration`, then exits. This gives a clean before/during/after structure in every episode, making the behavioral log's `shock_active` column interpretable as a controlled condition rather than random noise. In train mode, multiple overlapping shocks could fire at unpredictable times, which would make the shock-conditioned slice of the behavioral log harder for Module 4 to learn from.

**Fields set:**
- `eval_onset_time=3600.0` — shock fires one simulated hour into each episode
- `eval_shock_type='RESOURCE_REMOVAL'` — the most structured shock type (explicit resources removed)
- `eval_magnitude=0.3` — 30% of resources in the selected pool are removed
- `eval_duration=3600.0` — shock lasts one simulated hour

If you want to collect behavioral data under both shock types or different magnitudes, run the script twice with different `build_shock_config()` settings and concatenate the resulting CSVs before passing to Module 4.

---

## `collect(args)`

**Signature:** `collect(args: argparse.Namespace) -> None`

**Purpose:** The main collection function. Builds the environment, loads the model, runs episodes, accumulates rows, and writes the CSV.

---

### Inside `collect`

**`n_traces = args.n_traces if args.n_traces == 'from_input_data' else int(args.n_traces)`**

`parse_args` always stores `--n-traces` as a string (the default is `'from_input_data'`). This line converts it to an integer when it is a numeric string. The string `'from_input_data'` is passed through as-is because `gym_env.__init__` checks for that literal value and reads the trace count from the input JSON instead.

**`shock_config = build_shock_config() if args.shock else None`**

`args.shock` is `True` when `--shock` was passed on the CLI. If `None`, the `gym_env` does not register a shock SimPy process in `reset()` and no shocks fire. Collecting without shocks produces a behavioral log of the agent's baseline (no-shock) behaviour, which is useful for checking that the agent performs normally when not disrupted.

**`env = gym_env(..., postpone=False, reward_function='inverse_CT', ...)`**

Two constructor choices worth noting:
- `postpone=False` — Postpone is disabled so that every decision is a real (resource, activity) assignment. Postpone actions carry no tier information and would produce rows with no meaningful label for Module 4. Disabling it ensures every logged row is usable.
- `reward_function='inverse_CT'` — The reward is computed during `step()` but never used during collection. This value avoids the RF2 CPL-accumulation code path, which has no meaningful calibration during collection and would add unnecessary noise to `self.lagrange_lambda`.

**`model = MaskablePPO.load(args.model, env=env)`**

`MaskablePPO.load` reads the `.zip` file that SB3 saved at training time. The `env=env` argument is required to rebind the policy's internal environment reference; without it, `model.predict` cannot call `action_masks()`. SB3 appends `.zip` automatically when saving, and strips it automatically when loading — do not include it in `--model`.

**`all_cluster_keys = sorted({...})`**

This set comprehension iterates over the entire `resource_tier_map` and collects every `(activity, tier)` pair that exists anywhere in the map — regardless of whether that pair appears in any specific episode step. Sorting the set guarantees a deterministic column order across all rows and across runs. Without this pre-computation, different episodes might produce rows with different column sets, making `pd.DataFrame(rows)` fill in `NaN` rather than `0` for absent pairs.

**`col_names = [f"{act}|{tier}" for act, tier in all_cluster_keys]`**

The pipe character `|` is the separator between activity name and tier label in column headers. It was chosen because activity names in the target datasets use underscores heavily (`W_Assess_Application`) and dots are not valid in some CSV tools. A pipe produces readable column names like `W_Assess_Application|fast`.

---

### The episode loop

**`obs, _ = env.reset()`**

Resets the SimPy environment, spawns all tokens, advances to the first decision moment, and returns the initial individual-level observation vector. The second return value is an info dict (always empty here, per Gymnasium convention).

**`mask = env.action_masks()`**

Returns a list of booleans, one per entry in `env.output`. `True` means the action is currently valid (resource available and eligible for the pending activity). `MaskablePPO.predict` uses this to zero out logits for invalid actions before sampling.

**`action_idx, _ = model.predict(obs, deterministic=True, action_masks=mask)`**

`deterministic=True` selects the highest-probability action rather than sampling. This is standard practice for evaluation and collection — stochastic prediction would add variance across collection runs that does not reflect the true policy, only its temperature.

`action_idx` is returned as a numpy scalar. The explicit `int(action_idx)` cast on the next line converts it to a plain Python int for use as a list index.

**`chosen = env.output[action_idx]`**

`env.output` is the list of valid (resource, activity) pairs built from the artifact in `retrieve_possible_assignments()`. Index `action_idx` gives either a `(resource, activity)` tuple or the string `'Postpone'`. The `if chosen != 'Postpone':` guard skips logging for Postpone actions since they carry no tier assignment information.

**`tier_chosen = env.resource_tier_map.get(resource_chosen, {}).get(activity_chosen, 'unknown')`**

The double `.get(..., default)` chain is defensive: if somehow a resource has no entry in the tier map for the chosen activity (which should not happen given that eligibility is enforced by the artifact), `'unknown'` is returned rather than raising a `KeyError`. An `'unknown'` tier in the behavioral log would indicate a data consistency problem between the artifact and the model.

**`cluster_obs = env.get_cluster_obs()`**

Calls the utility method added to `gym_env` in Step 18. Returns a dict `{(activity, tier): count}` reflecting which resources are currently available. Crucially, this is called *before* `env.step(action_idx)` — the cluster observation must reflect the state *at the moment the agent made its decision*, not after the environment has advanced.

**Row construction:**
```python
row = {
    'episode':          episode,
    'step':             step_in_ep,
    'pending_activity': activity_chosen,
    'shock_active':     int(env.simulation_process.shock_state.is_active),
    'tier_chosen':      tier_chosen,
}
for key, col in zip(all_cluster_keys, col_names):
    row[col] = cluster_obs.get(key, 0)
```
The `zip(all_cluster_keys, col_names)` pairs each `(activity, tier)` tuple with its pre-computed column name string. `cluster_obs.get(key, 0)` returns `0` for pairs that exist in the map but happen to have no available resources at this step — this is different from a pair that does not exist at all in `resource_tier_map`, which is simply never a key in `all_cluster_keys`.

**`obs, _reward, done, _, _ = env.step(action_idx)`**

Steps the environment forward. The five return values are: next observation, reward, terminated flag, truncated flag (always `False` here), info dict (always empty). Only `obs` and `done` are used. The reward is discarded because it plays no role in collection.

---

### Post-loop

**`df = pd.DataFrame(rows)`**

Converts the list of dicts into a DataFrame. Because every dict was built with the same keys (including all `all_cluster_keys` columns), there are no `NaN` values — every cell is either a real count or an explicit `0`.

**`os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)`**

Creates the output directory if it does not exist. `os.path.abspath` resolves relative paths (e.g., `../Module 1/artifacts/`) to absolute paths before `os.path.dirname` extracts the directory portion. `exist_ok=True` suppresses the error if the directory already exists.

**`df.to_csv(args.output, index=False)`**

Writes the DataFrame to CSV without the pandas row index (which would add a meaningless integer column). The resulting file has one header row and one data row per non-Postpone decision.

**Summary print block**

Prints total decision count, episode count, column list, and tier distribution to stdout. The tier distribution (`df['tier_chosen'].value_counts()`) is the most useful diagnostic: if one tier dominates heavily (e.g., 95% fast-tier), the agent has learned a near-deterministic policy, which may mean fewer distinct rules in the Module 4 decision trees. If the distribution is roughly uniform, the agent varies its allocation strategy meaningfully across conditions.

---

## Output CSV schema

| Column | Type | Description |
|---|---|---|
| `episode` | int | Which collection episode (0-indexed) |
| `step` | int | Decision number within the episode (0-indexed) |
| `pending_activity` | str | The activity that needed a resource at this step |
| `shock_active` | int (0/1) | Whether a shock was active at this moment |
| `tier_chosen` | str | The tier of the resource the agent assigned (`slow`, `mid`, `fast`, or `unknown`) |
| `<act>\|<tier>` | int | Count of available resources with tier `<tier>` for activity `<act>` at this moment |

There is one `<act>|<tier>` column for every (activity, tier) pair present in `resource_tier_map`, regardless of whether that pair was observed in any given step. Absent pairs are zero-filled.

---

## Module 4 usage

Module 4 reads this CSV, splits by `pending_activity`, and trains one decision tree per activity:

```
features: shock_active, all <act>|<tier> columns
label:    tier_chosen
```

Each path through the tree becomes one rule in a DMN decision table or one guard condition on an XOR gateway in BPMN. The `episode` and `step` columns are for traceability and are dropped before training the decision trees.
