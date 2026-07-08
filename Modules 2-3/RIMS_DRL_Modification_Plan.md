# RIMS_DRL Modification Record

This document records every change made to the `RIMS_DRL-main` codebase, in execution order. All 19 steps are complete. Each entry names the file, what was changed, and why.

---

## Phase 1 — Unblock the code (remove LSTM/Prophet dependencies)

Steps 1–4 make the code runnable without TensorFlow, Prophet, or any `.h5` model file.

### Step 1 — `parameters.py`: METADATA fallback

**Why:** `read_metadata_file()` crashes when the metadata `.json` does not exist because `RESOURCE_TO_ROLE_LSTM` is never set, and `gym_env` reads it immediately on construction.

**Done:** Added an `else` branch to the `if os.path.exists(self.METADATA):` block. When the file is absent, `RESOURCE_TO_ROLE_LSTM` is built from `data['resource']` in the input JSON — each resource name maps to itself as its own role. All other METADATA fields (`INDEX_AC`, `AC_WIP_INITIAL`, etc.) are LSTM-only and need no fallback after Step 2 removes the predictor.

### Step 2 — `process.py`: Remove the LSTM Predictor

**Why:** `call_LSTM.py` requires TensorFlow and pre-trained `.h5` files that are not part of this thesis setup.

**Done:** Removed `from call_LSTM import Predictor`, deleted `self.predictor = Predictor(...)` and `self.predictor.predict()` from `__init__`, and deleted the `get_predict_processing` method entirely. Added `self.resource_processing_times = {}` as a placeholder (populated by `gym_env` in Step 6).

### Step 3 — `inter_trigger_timer.py`: Remove the Prophet dependency

**Why:** `__init__` unconditionally calls `generate_arrivals`, which crashes for the `distribution` type because `self.params` is reassigned to a plain dict before the call, and `generate_arrivals` immediately accesses `self.params.MODEL_PROPHET`.

**Done:** Guarded `generate_arrivals` so it only runs when `self._type == 'custom'`. Made the Prophet import optional with `try/except` and an `_PROPHET_AVAILABLE` flag. Added `arrival_multiplier` division in `get_next_arrival` (hook for DEMAND_SPIKE, completed in Step 12).

### Step 4 — `event_trace.py`: Replace the LSTM duration call

**Why:** The original lines 139–148 call `get_predict_processing` (removed in Step 2) and read LSTM-specific buffer features that no longer exist.

**Done:** Replaced the six LSTM lines with an empirical lookup: `duration = pt_map[res_name][act_name]`, where `pt_map` is `self._process.resource_processing_times`. Added a global-mean fallback with a warning for the defensive case where a pair is missing. Removed the LSTM-only buffer feature lines (`ro_single`, `ro_total`, `role` set twice).

---

## Phase 2 — Wire in the Module 1 artifact

Steps 5–7 connect the empirical artifact produced by `empirical_preprocessing.py` into the simulation.

### Step 5 — `gym_env.py`: Load the Module 1 artifact at startup

**Done:** Added `artifact_path=None` parameter to `__init__`. Raises `ValueError` if not provided. Loads the JSON and stores `artifact['resource_processing_times']` as `self.resource_processing_times` and `artifact['resource_tier_map']` as `self.resource_tier_map`.

### Step 6 — `process.py`: Expose `resource_processing_times` on `SimulationProcess`

**Why:** `event_trace.py` accesses `self._process.resource_processing_times` after Step 4. The attribute must exist on `SimulationProcess` at construction time even though `gym_env` fills it immediately after.

**Done:** Added `self.resource_processing_times = {}` to `SimulationProcess.__init__`. Set by `gym_env.reset()`: `self.simulation_process.resource_processing_times = self.resource_processing_times`.

### Step 7 — `gym_env.py`: Rewrite `retrieve_possible_assignments` from artifact

**Why:** The original method read the XES log at every call and returned all observed (resource, activity) pairs. After Step 4, eligibility is defined by the artifact — a pair is eligible if and only if it has an empirical duration that passed Module 1's n_min filter (D4).

**Done:** Rewrote the method to build the list from `self.resource_processing_times` keys. The XES log is no longer read. Postpone logic unchanged.

---

## Phase 3 — D10: Uniform calendars

### Step 8 — `train_DRL.py`: Set `CALENDAR=False`

**Why:** D10 specifies uniform calendars for all resilience experiments to eliminate off-hours waiting from the simulation, isolating the effect of the agent's allocation policy.

**Done:** Set `CALENDAR = False` as the default in `train_DRL.py`. When `False`, `event_trace.py` skips the `to_time_schedule` check entirely, so no off-hours delay is added.

---

## Phase 4 — D5: Shock injection

Steps 9–13 add the shock layer. Two shock types: `RESOURCE_REMOVAL` (fraction of resources become unavailable) and `DEMAND_SPIKE` (arrival rate multiplied).

### Step 9 — New file `shock.py`

**Done:** Created `shock.py` containing:
- `ShockConfig` (dataclass): holds all shock parameters for train mode (Poisson rate, magnitude range, duration range, shock types, resource groups) and eval mode (fixed onset time, type, magnitude, duration).
- `ShockState` (class): mutable per-episode runtime state — `is_active`, `active_type`, `active_magnitude`, `removed_resources` (set).
- `shock_scheduler` (SimPy generator function): in train mode, loops drawing Poisson inter-shock intervals and firing random shocks; in eval mode, fires exactly one shock and exits.
- `_apply_onset` / `_apply_offset` (helpers): update `ShockState` and call `mark_removed`/`mark_restored` on `RoleSimulator` instances (RESOURCE_REMOVAL) or set/reset `arrival_multiplier` on `SimulationProcess` (DEMAND_SPIKE).

### Step 10 — `role_simulator.py`: Add `removed` flag

**Why:** SimPy `Resource` capacity is immutable at runtime. Removal is implemented as a flag that the availability check reads, rather than as a capacity change.

**Done:** Added `self.removed = False` to `__init__`. Added `mark_removed()` (sets flag `True`) and `mark_restored()` (sets flag `False`). Busy resources are not pre-empted — they finish their current task naturally.

### Step 11 — `process.py`: Integrate shock layer

**Done:** Added import `from shock import ShockState`. Added three attributes to `SimulationProcess.__init__`: `self.shock_state = ShockState()`, `self.arrival_multiplier = 1.0`, `self.shock_config = None`. Added `start_shock_scheduler(env)` method that registers the `shock_scheduler` SimPy process. Updated `get_state()` to skip shock-removed resources into `resource_anvailable` so availability checks are automatically correct downstream.

### Step 12 — `inter_trigger_timer.py`: Apply arrival multiplier

**Why:** DEMAND_SPIKE sets `simulation_process.arrival_multiplier > 1.0`. The inter-arrival time must be divided by this value to compress the interval and increase the effective arrival rate.

**Done:** In `get_next_arrival`, after drawing `arrival` from the distribution, added `arrival = arrival / self._process.arrival_multiplier`. This was already scaffolded in Step 3; Step 12 ensures the attribute exists on `SimulationProcess` (Step 11) and reads correctly.

### Step 13 — `gym_env.py`: Pass `shock_config`; start scheduler in `reset()`

**Done:** Added `shock_config=None` parameter to `__init__`, stored as `self.shock_config`. In `reset()`, after constructing `SimulationProcess`, if `shock_config` is not `None`: sets `self.simulation_process.shock_config = self.shock_config` and calls `self.simulation_process.start_shock_scheduler(self.env)`. Removed resources are excluded from `resource_available` automatically via `process.get_state()` (Step 11), so `get_state()` and `action_masks()` in `gym_env` are shock-aware without additional changes.

---

## Phase 5 — D7: Extend the observation vector

### Step 14 — `gym_env.py`: Add `shock_active` to the observation

**Why:** Without the shock flag, the agent cannot distinguish shock from no-shock states and cannot condition its policy on disruption.

**Done:** Appended `'shock_active'` to `self.input` in `__init__`. Appended `float(self.simulation_process.shock_state.is_active)` to the returned array in `get_state()`. The observation space `lows`/`highs` arrays are built from `len(self.input)` and expand automatically.

---

## Phase 6 — D6: RF2 reward

### Step 15 — `gym_env.py`: CPL tracking infrastructure

**Why:** RF2 requires per-episode tracking of cumulative process lag (CPL) and a Lagrange multiplier `u_k` that is updated at episode end and persists across episodes.

**Done:** Added `cpl_budget=0.0` and `eta_lambda=0.01` constructor parameters. Added `self.lagrange_lambda = 0.0` at constructor level (persists across episodes). In `reset()`, initialized `self.cpl_accumulated = 0.0` and `self.prev_env_time = 0.0` (reset each episode). Appended `'u_k'` to `self.input` and `min(1.0, self.lagrange_lambda)` to the `get_state()` return (clamped to [0,1] for observation stability).

### Step 16 — `gym_env.py`: RF2 reward branch

**Done:** In `step()`, added RF2 handling alongside the existing `inverse_CT` / `negative_CT` branches:
- Base reward: same normalized inverse cycle time as `inverse_CT` for completed traces.
- CPL accumulation: when shock is active, `cpl_delta = dt * n_pending / normalization_cycle_time` is added to `self.cpl_accumulated` and subtracted from reward scaled by `self.lagrange_lambda`.
- `prev_env_time` updated each step.
- At episode end: `self.lagrange_lambda = max(0.0, self.lagrange_lambda + eta_lambda * (cpl_accumulated - cpl_budget))`.

---

## Phase 7 — Wire everything together

### Step 17 — `train_DRL.py`: Update entry point

**Done:**
- Added `artifact_path` as the 8th CLI argument (required); must point to the Module 1 JSON artifact.
- Added `ShockConfig` construction block at module level (edit fields there before running).
- Added FIFO calibration run before training when `reward_function == 'RF2'`: runs one episode with first-valid-action FIFO policy, records `cal_env.cpl_accumulated` as `cpl_budget`.
- Updated `log_dir` naming to include `_S0`/`_S1` shock tag.
- Added `n_steps.get(NAME_LOG, 5120)` fallback for datasets not in the original table.
- Both `env_simulator` and `eval_env` receive `artifact_path`, `shock_config`, `cpl_budget`, `eta_lambda`.

---

## Phase 8 — Behavioral log collection (post-training, feeds Module 4)

These two steps run after training is complete. They are not part of the training loop.

### Step 18 — `gym_env.py`: Add `get_cluster_obs()`

**Why:** Module 4 distillation operates on cluster-level (tier-level) data. `get_cluster_obs()` translates the live simulation state into a count of available resources per (activity, tier) pair. It is a read-only utility — it has no effect on the agent, the reward, or any training logic.

**Done:** Added `get_cluster_obs()` method to `gym_env`, placed before `retrieve_possible_assignments()`. Iterates over `self.resource_tier_map`, checks each resource against `process.get_state()['resource_available']` (which already excludes shock-removed resources per Step 11), and returns `{(activity, tier): count}`. Never called during training.

### Step 19 — New file `collect_behavioral_log.py`

**Why:** A standalone script with a frozen model produces a clean behavioral log of the final policy. Collecting during training would mix decisions from many policy versions.

**Done:** Created `collect_behavioral_log.py`. Accepts CLI args (`--name-log`, `--n-traces`, `--artifact`, `--model`, `--output`, `--n-episodes`, `--shock`, `--threshold`). Builds `gym_env` with `postpone=False` (every step is a real decision) and `reward_function='inverse_CT'` (reward discarded). Loads frozen `MaskablePPO` model and runs `--n-episodes` episodes with `model.predict(deterministic=True)`. At each non-Postpone step: calls `get_cluster_obs()`, translates agent's resource choice to a tier via `resource_tier_map`, appends a row. Writes flat CSV with fixed columns (all (activity, tier) pairs from `resource_tier_map`, zero-filled when absent). See `collect_behavioral_log_technical.md` for full variable-level documentation.

**Example call:**
```bash
python collect_behavioral_log.py \
    --name-log bpi12w \
    --n-traces 500 \
    --artifact "../Module 1/artifacts/bpi12w.json" \
    --model "tmp_training_2/bpi12w_.../model_final" \
    --output "../Module 1/artifacts/bpi12w_behavioral_log.csv" \
    --n-episodes 50 \
    --shock
```

---

## Execution order summary

| Step | File | What was done |
|------|------|---------------|
| 1 | `parameters.py` | Added `else` branch: build `RESOURCE_TO_ROLE_LSTM` from input JSON when METADATA file absent |
| 2 | `process.py` | Removed Predictor import, instantiation, and `get_predict_processing` method; added `resource_processing_times = {}` |
| 3 | `inter_trigger_timer.py` | Guarded `generate_arrivals` for `distribution` type; made Prophet import optional; scaffolded `arrival_multiplier` hook |
| 4 | `event_trace.py` | Replaced LSTM duration call with empirical lookup + global-mean fallback; removed LSTM-only buffer features |
| 5 | `gym_env.py` | Added `artifact_path` param; load Module 1 JSON; stored `resource_processing_times` and `resource_tier_map` |
| 6 | `process.py` | Added `resource_processing_times = {}` to `SimulationProcess.__init__`; set by `gym_env.reset()` |
| 7 | `gym_env.py` | Rewrote `retrieve_possible_assignments` to build from artifact instead of XES log |
| 8 | `train_DRL.py` | Set `CALENDAR = False` as default |
| 9 | new `shock.py` | Created `ShockConfig`, `ShockState`, `shock_scheduler`, `_apply_onset`, `_apply_offset` |
| 10 | `role_simulator.py` | Added `removed = False`; added `mark_removed()` and `mark_restored()` |
| 11 | `process.py` | Added `shock_state`, `arrival_multiplier`, `shock_config`; added `start_shock_scheduler()`; updated `get_state()` to exclude removed resources |
| 12 | `inter_trigger_timer.py` | Added `arrival = arrival / self._process.arrival_multiplier` in `get_next_arrival` |
| 13 | `gym_env.py` | Added `shock_config` param; call `start_shock_scheduler` in `reset()` when config is set |
| 14 | `gym_env.py` | Appended `shock_active` to `self.input` and `get_state()` return |
| 15 | `gym_env.py` | Added `cpl_budget`, `eta_lambda` params; `lagrange_lambda` persists across episodes; `cpl_accumulated`, `prev_env_time` reset each episode; appended `u_k` to observation |
| 16 | `gym_env.py` | Added RF2 reward branch: inverse-CT base + CPL accumulation during shock + Lagrange update at episode end |
| 17 | `train_DRL.py` | Added `artifact_path` CLI arg, `ShockConfig` block, FIFO CPL calibration, shock-tagged `log_dir`, `n_steps` fallback |
| 18 | `gym_env.py` | Added `get_cluster_obs()` — returns `{(activity, tier): count}` from live sim state; never called during training |
| 19 | new `collect_behavioral_log.py` | Post-training script: loads frozen model, runs collection episodes, writes behavioral log CSV for Module 4 |

**Milestones:**
- Steps 1–4: code runs without TensorFlow/Prophet
- Steps 5–8: DRL training loop works with empirical durations
- Steps 9–13: shock injection active
- Steps 14–16: extended observation and RF2 reward complete
- Step 17: full training harness wired
- Steps 18–19: Module 4 feed complete (post-training only)
