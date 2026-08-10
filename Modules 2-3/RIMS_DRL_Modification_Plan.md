# RIMS_DRL Modification Record

This document records every change made to the `RIMS_DRL-main` codebase, in execution order. All 59 steps are complete (Phases 17 and 20 list validated-but-not-yet-integrated work — see those sections; Phase 21 is where that methodology finally gets wired into production code for real). Consulta and BPI2017 are calibration-complete, training-ready, and have passed an end-to-end training smoke test with both shock types active as of Step 55; `phi(t)` was subsequently redefined as a non-overlapping tumbling-batch mean (Steps 56-57) and both datasets recalibrated and re-validated against it (Steps 58-59), with resilience-metrics results now showing dramatically tighter seed variance and clear, dataset-specific dose-response patterns. Helpdesk has a validated, citation-backed duration calibration and a corrected `P*` (Steps 47-50) but is not yet training-ready — its shock calibration remains open (see `Training_Readiness_Implementation_Summary.md` §45). Each entry names the file, what was changed, and why.

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

## Phase 9 — Module 1: artifact schema extension (P*, C, PPI window, placeholder resource, wide-format PT, Consulta fix)

Brings `empirical_preprocessing.py`'s output up to what RF1/RF2 need, and fixes two gaps
found during dataset triage (`Module 1/Dataset_Suitability_Assessment.md`): zero-eligible-
resource activities silently deadlocking the simulator, and Consulta's undetected
wide-format processing times / fake-resource activities.

### Step 20 — `empirical_preprocessing.py`: wide-format true-PT detection

**Why:** `load_and_prepare()` only recognised true start→complete durations as two separate
rows with different `lifecycle` values. ConsultaDataMining201618 carries a `start:timestamp`
attribute on the *same* row as the completion timestamp (`lifecycle:transition` uniformly
`"complete"`), so it silently fell back to the cruder sojourn approximation despite better
data being available.

**Done:** `load_and_prepare()` now also detects and preserves a `start_timestamp` column
(`start:timestamp` for XES, common aliases for CSV) and returns a 3-tuple
`(df, has_true_pt_long, has_true_pt_wide)`. Added `_compute_true_pt_wide()`: per-row
`pt = timestamp - start_timestamp`, no cross-row pairing needed. `run()`'s precedence is
now long-format true PT > wide-format true PT > sojourn approximation. `compute_p_star.py`
updated for the new 3-tuple return (unaffected otherwise — P* only uses case timestamp span).

### Step 21 — `empirical_preprocessing.py`: activity-name whitespace stripping

**Why:** Found during Phase 14 validation. BPI2017's raw log carries the activity name
`'W_Shortened completion '` with a trailing space (`concept:name`, 238 events), but the
corresponding mined Petri net's task name is `'W_Shortened completion'` (no trailing space).
Module 1's artifact was keyed by the untrimmed name, so the placeholder-resource fix (Step 22)
never actually matched what the simulator asked about — this looked identical to a genuine
zero-eligible-resource activity but wasn't one, and caused a real (multi-minute) hang in
`next_decision_moment()` once the agent's only pending token needed that exact activity.

**Done:** Added `df["activity"] = df["activity"].str.strip()` in `load_and_prepare()`,
right after the format-specific load and before activity merging/lifecycle detection.

### Step 22 — `empirical_preprocessing.py`: activity merge/exclude + synthetic placeholder resource

**Why:** Two problems needed a shared mechanism: (a) any activity left with zero eligible
resources after Module 1's filters has no valid `(resource, activity)` pair anywhere
downstream, so its action mask is permanently empty; (b) Consulta's
`"Traer informacion estudiante - banner"` / `"Radicar Solicitud Homologacion"` have
`resource` = the student's own account ID (99% self-submission), not real staff — need to
be merged into one activity and then excluded from resource eligibility entirely, which
routes them through fix (a).

**Done:**
- Added `activity_merge_map` param to `load_and_prepare()`/`run()` — renames activities
  right after column normalisation (generic, not Consulta-specific).
- Added `exclude_activities` param to `run()` — strips named activities out of
  `resource_pt`/`support_counts` after `aggregate()`, before `validate()`, so they land in
  `activities_at_risk`.
- After `validate()` computes `activities_at_risk`, `run()` now injects a synthetic
  `"__NO_RESOURCE__"` key into `resource_processing_times` for every at-risk activity
  (duration = dataset-wide median processing time across all eligible pairs), added
  **after** `compute_tiers()` so it's never part of `resource_tier_map` — it must never enter
  the cluster-layer observation. New CLI flags: `--merge-activities OLD=NEW ...`,
  `--exclude-activities ACT ...`.

### Step 23 — `empirical_preprocessing.py`: `P_star` / `normalization_cycle_time` / `ppi_window_n` fields

**Why:** RF1/RF2 both need `P*` (the acceptable-performance threshold) and a rolling-PPI
window size at runtime; RF2 also needs the `C` scaling constant. These previously had nowhere
to live — deferred earlier in the project ("decide about JSON extension later").

**Done:** Added `p_star`, `normalization_cycle_time` (default `10000.0`), `ppi_window_n`
(default `20`) params to `run()`, written directly into the artifact JSON alongside
`resource_processing_times`. `P*` itself is **not** recomputed here — it's still the p95 of
per-case lead time from the existing `compute_p_star.py` (unchanged method), merged in via
these new params. `normalization_cycle_time` is set to the literal `10000` for every dataset
(Meneghello et al.'s original value, not derived from data), with a `_normalization_cycle_time_note`
field flagging it as not yet re-tuned per dataset. New CLI flags: `--p-star`,
`--normalization-cycle-time`, `--ppi-window-n`. All four artifacts
(`bpi2017.json`, `bpi12w.json`, `helpdesk.json`, and the first-ever `consulta.json`)
regenerated with these fields plus Steps 20-22's fixes.

---

## Phase 10 — Modules 2-3: shared plumbing (rolling PPI, artifact fields, postpone removal, placeholder wiring)

### Step 24 — `gym_env.py`: rolling PPI `phi(t)` tracking

**Done:** Added `self.ppi_history` (`collections.deque`, `maxlen=self.ppi_window_n`),
reset every episode in `reset()`. Updated inside the existing per-trace "gather rewards"
loop in `step()` — `self.ppi_history.append(cycle_time)` for every newly-ended trace,
regardless of `reward_function` (cheap, shared infra). Added `self.prev_ppi` (tracks
`phi(t-1)` for RF1) and a `_current_ppi()` helper (`mean(ppi_history)`, `0.0` if empty).

### Step 25 — `gym_env.py`: load `P_star`/`ppi_window_n`/`normalization_cycle_time` from artifact

**Done:** Extended the Step 5 artifact-loading block to also read `artifact.get('P_star')`,
`artifact.get('ppi_window_n', 20)`, and `artifact.get('normalization_cycle_time', ...)` —
the last one now overrides the previously-hardcoded `self.normalization_cycle_time = 10000`
with the artifact-sourced (same-valued, but now per-dataset-tunable) value. Also stores
`self.activities_at_risk = artifact.get('activities_at_risk', [])` for Step 27.

### Step 26 — `train_DRL.py`: disable postpone everywhere

**Why:** The current Action Space spec (`Module_3_DRL_Environment_and_Agent.md` §2) has no
no-op/postpone action — the agent always assigns a resource to the pending activity. The
`postpone` CLI arg/constructor default (`True`) predates this and no longer applies.

**Done:** `sys.argv[5]` is still parsed (CLI positional-arg compatibility) but its value is
now ignored — every `gym_env(...)` construction site in `train_DRL.py` (training env, eval
env, both calibration envs) hardcodes `postpone=False`.

### Step 27 — `gym_env.py`/`process.py`: synthetic placeholder resource wiring

**Why:** `resource_processing_times` having a `"__NO_RESOURCE__"` entry (Step 22) isn't
enough by itself — `self.resources` (built from `input_data["resource"].keys()`, a
different source entirely) and `SimulationProcess._resources` (built from
`Parameters.ROLE_CAPACITY`, yet another source) both need the placeholder registered too,
or the observation/action-space indexing and the actual SimPy resource object won't exist.

**Done:**
- `gym_env.__init__`: after building `self.resources`, appends `PLACEHOLDER_RESOURCE`
  (`"__NO_RESOURCE__"`) if `self.activities_at_risk` is non-empty and it isn't already
  present — done once, since the action/observation space size is fixed at construction.
- `gym_env.reset()`: after constructing `SimulationProcess`, registers
  `PLACEHOLDER_RESOURCE` as a `RoleSimulator` with `capacity=math.inf` (always available) —
  done every episode, since `SimulationProcess` is rebuilt fresh each `reset()`.
- Added `PLACEHOLDER_RESOURCE`/`_ALWAYS_ON_CALENDAR` module-level constants and a
  `from role_simulator import RoleSimulator` import to `gym_env.py`.
- Confirmed by construction (not just by inspection) that `"__NO_RESOURCE__"` can never
  appear as a valid action for any activity that already has real eligible resources — it
  only has an entry in `resource_processing_times` for the specific at-risk activities.

---

## Phase 11 — Calibration: `phi_fix` (RF1) and `CPL_base` (RF2)

### Step 28 — `train_DRL.py`: fix the FIFO `cal_env` bug

**Why:** `cal_env` was constructed with `reward_function='inverse_CT'`, but
`self.cpl_accumulated` is only ever incremented inside `gym_env.step()`'s
`if self.reward_function == 'RF2':` branch. `cal_env.cpl_accumulated` therefore stayed at
its `reset()`-time value of `0.0` for the entire calibration episode — `cpl_budget` was
always `0.0` regardless of what the FIFO policy actually did.

**Done:** `cal_env` now constructed with `reward_function='RF2'` instead. This does **not**
change the action-selection policy (`action = valid[0]`, still FIFO/first-eligible-action) —
`reward_function` only controls which internal bookkeeping branches execute in `step()`.

### Step 29 — `train_DRL.py`: `p` budget fraction

**Why:** Spec: `CPL_max = p * CPL_base`, `p ∈ (0,1]` — previously absent entirely
(`cpl_budget = cal_env.cpl_accumulated` used the raw value, i.e. implicitly `p=1.0`).

**Done:** Added `p_cpl_fraction = 0.5` (spec's own example value) at module level;
`cpl_budget = p_cpl_fraction * cal_env.cpl_accumulated`.

### Step 30 — `train_DRL.py`: random-policy `phi_fix` calibration loop

**Why:** RF1's `phi_fix` (disrupted-minimum PPI anchor) has no prior calibration mechanism
in this codebase — the project's very first ask was "we have not... run a simulation with
random policy," and this is that mechanism's actual purpose.

**Done:** When `reward_function == 'RF1'` and `shock_config is not None`, runs
`K_PHI_FIX_ROLLOUTS = 5` episodes with a uniformly-random *valid* action each step
(`np.random.choice(valid)`), tracking the worst (max) `env._current_ppi()` observed per
episode via `rand_env._current_ppi()`. `phi_fix = max` across all `K` episodes — the spec's
stated default calibration method (§8: "worst PPI observed over a set of random-policy
rollouts").

---

## Phase 12 — Implement Reward Function 1 (RF1)

### Step 31 — `gym_env.py`: `phi_fix`/`beta` params + RF1 branch

**Why:** RF1 did not exist in code at all prior to this — selecting `'RF1'` would have
silently produced all-zero reward (no matching branch anywhere in `step()`).

**Done:** Added `phi_fix=None`, `beta=0.5` constructor params. Added an `'RF1'` branch inside
the existing per-trace "gather rewards" loop in `step()`, evaluated once per newly-ended
trace (mirroring where `phi(t)` changes):
```python
denom = abs(self.P_star - self.phi_fix) or 1e-6
phi_t = self._current_ppi()
r1 = (self.prev_ppi - phi_t) / denom          # velocity (potential-based shaping)
r2 = min(0.0, -(phi_t - self.P_star) / denom)  # position (deviation penalty)
reward += self.beta * r1 + (1 - self.beta) * r2
self.prev_ppi = phi_t
```
Raises `ValueError` if `phi_fix`/`P_star` aren't set when RF1 is selected, rather than
computing on `None`. No `gamma` discount correction applied to the shaping term (spec calls
this negligible at `gamma≈0.99` and optional).

---

## Phase 13 — Rewrite Reward Function 2 (RF2) to spec

### Step 32 — `gym_env.py`: `d(t)`-based CPL accumulation, `u_k`/`v_k`, Lagrange update, observation fix

**Why:** The existing RF2 implementation used `n_pending` (pending-token count) as a proxy
for performance loss, gated strictly on `shock_state.is_active`, updated `lambda` from the
raw `cpl_accumulated - cpl_budget` difference, and exposed `lambda` itself in the
observation mislabeled as `'u_k'`. None of these match the SMDP/constrained-MDP formulation
in the updated `Module_3_DRL_Environment_and_Agent.md` §5.

**Done, all in `gym_env.py`:**
- CPL accumulation replaced: `d_t = max(0.0, phi_t - self.P_star)`,
  `self.cpl_accumulated += d_t * dt` (`B_k`). The `shock_state.is_active` gate is dropped —
  `d(t)` is already ~0 whenever performance is at/below `P*`, so this naturally keeps
  accruing through the post-shock recovery tail instead of stopping the instant the shock
  flag clears.
- `self.u_k = cpl_accumulated / cpl_budget` (`0.0` if `cpl_budget<=0`),
  `self.v_k = max(0.0, u_k - 1.0)`, computed every step.
- Reward term: `reward -= self.lagrange_lambda * self.v_k` — applied unconditionally every
  step (`f2=1` always; there's no postpone/no-op action per Step 26, so every decision step
  is "the agent taking an action").
- Episode-end Lagrange update changed to `lagrange_lambda = max(0, lagrange_lambda +
  eta_lambda * v_k)` (spec's violation-signal-driven update), replacing the old
  `cpl_accumulated - cpl_budget` raw-difference update.
- Observation fix: the `'u_k'` feature in `get_state()` previously returned
  `min(1.0, self.lagrange_lambda)` — i.e. `lambda`, not `u_k`. Now returns
  `min(1.0, self.u_k)` (clamped to `[0,1]` to match the declared `observation_space` Box
  bounds — the violation itself is still visible to the agent via the reward penalty).
  **This only changes what value is written into that one observation slot — it does not
  touch `v_k`'s definition, `lambda`'s update rule, or the reward term above.**
- Also fixed a related crash: the episode-end summary block never set
  `self.total_reward_end` for `reward_function == 'RF1'` (only `inverse_CT`/`RF2`/`negative_CT`
  had branches), crashing on the unconditional print at episode end. Added an `'RF1'` branch
  that mirrors `self.total_reward` (RF1's reward isn't a pure post-hoc function of final
  cycle times the way the others are, so there's no independent "final calc" to recompute).

---

## Phase 14 — Validation pilot: BPI2017 (instrumentation only, no training)

Manual environment stepping (FIFO/random policies scripted externally, no `model.learn()`
anywhere) confirming Phases 9-13 behave sanely before rolling out to BPI12W/Helpdesk/Consulta.
All checks pass as of this writing:
- Artifact fields (`P_star`, `normalization_cycle_time`, `ppi_window_n`, placeholder entries)
  present and correctly scoped (`"__NO_RESOURCE__"` absent from `resource_tier_map`).
- A scripted FIFO episode that reaches a previously-at-risk activity confirms the action
  mask is no longer permanently empty there.
- FIFO `CPL_base` and 3x random-policy `phi_fix` calibration both produce finite, non-zero
  values; `phi_fix (5,046,268.7s) > P_star (3,673,816.2s)` — a random policy performing worse
  than the historical acceptable level, as expected.
- RF1 manual stepping: reward finite and non-degenerate (142/886 nonzero steps).
- RF2 manual stepping: `u_k`/`v_k`/`lambda` evolve sensibly; observation's `u_k` feature
  confirmed to match the tracked `u_k`, not `lambda`.

**Two bugs found and fixed during this pass** (Steps 21 and the `total_reward_end` fix
folded into Step 32) — see those entries above.

**One incidental fix, unrelated to reward logic:** `inter_trigger_timer.py`'s
`generate_arrivals()` (Prophet-based arrival forecasting for `interTriggerTimer.type ==
'custom'`, used by BPI2017/BPI12W/Consulta) crashed under the currently-installed
pandas/prophet versions — the serialized model JSON encodes UTC timestamps with a trailing
`Z` that newer pandas' `read_json(orient='table')` rejects when the schema doesn't also
declare a `tz`. Fixed with a regex strip of the trailing `Z` before deserializing (the
timestamps were never meaningfully tz-aware). Unrelated to Module 1/RF1/RF2, but blocked all
validation until patched.

**Known open issue, not fixed here:** `next_decision_moment()` (`gym_env.py`) has no bound on
how long it searches for a valid decision. With shocks enabled, a `RESOURCE_REMOVAL` shock
can disable enough of an activity's (already finite) resource pool that the search stalls for
real minutes before self-resolving — confirmed at both 25 and 150 traces. This is independent
of `postpone` (the function never reads that flag) and predates all of Phases 9-14; it simply
had never been exercised end-to-end with shocks forced to run to completion before. Phase 14's
RF1/RF2 checks above were run with shocks disabled to get a clean read on the reward math
itself. **Needs a decision before shock-enabled training runs can be trusted at scale** — e.g.
an iteration/time bound on `next_decision_moment()` with a defined fallback.

---

## Phase 15 — Resolve Step 34's livelock risk: single-shock-per-episode redesign

### Step 35 — `shock.py` / `gym_env.py` / `train_DRL.py`: single-shock-per-episode redesign

**Why:** Step 34 flagged but didn't fix a livelock risk: `shock_scheduler`'s train-mode Poisson
loop never stops firing shocks, so `env.peek()` can never return `math.inf` —
`next_decision_moment()`'s only exit condition becomes "a resource-availability change just
unblocked a pending token," which a `RESOURCE_REMOVAL` shock hitting an already-thin per-activity
pool could leave unsatisfied for real minutes. This is independent of `postpone` (the function
never reads that flag).

**Done:**
- `shock.py`: added `fire_single_shock()` — samples type/magnitude/duration the same way
  `shock_scheduler` already did, applies the onset once, schedules exactly one delayed offset
  via a single `env.process(...)`, no re-arming. Also excluded `PLACEHOLDER_RESOURCE` from the
  `RESOURCE_REMOVAL` candidate pool (it was previously eligible for removal, which would have
  disabled the one resource meant to always be available).
- `gym_env.py`: `reset()` no longer calls `start_shock_scheduler()`. Instead, a hidden FIFO
  warm-up loop (reusing `step()`'s own assignment machinery, reward discarded) runs until
  `ppi_history` is full (`ppi_window_n` completed traces), establishing a real `phi(t)` baseline
  before `fire_single_shock()` fires — the agent's first-ever observation coincides exactly with
  the shock's onset instant (verified by assertion). `total_reward`/`reward_count`/
  `cpl_accumulated`/`prev_env_time` are zeroed after warm-up so bookkeeping reflects the shock
  episode, not the warm-up; `ppi_history`/`prev_ppi` are deliberately *not* reset (the
  established baseline carries forward). `step()`'s episode-end block now computes
  recovery-based termination — `terminated` when either all tokens are done or
  (`shock_state.is_active` just became `False` **and** `phi(t) <= P_star`) — plus a backstop
  `truncated` (`5 * duration_range[1]` after shock end) using Gymnasium's real `truncated` slot
  (previously hardcoded `{}` regardless of outcome).
- `train_DRL.py`: both calibration loops (FIFO `cal_env`, random-policy `phi_fix` rollouts)
  updated to exit on `terminated or truncated`, not just `terminated`.

**Validated:** with shocks *enabled* (unlike Step 34's validation, which ran shocks disabled to
sidestep this exact issue) — `env.now == shock_onset_time` held at every reset, no hang at any
scale tested, every episode terminated via recovery.

---

## Phase 16 — Reliability fixes found during cross-dataset validation

### Step 36 — `gym_env.py`: Consulta activity-merge-map wiring

**Why:** Module 1 (Step 22) merges two raw activity labels
(`"Traer informacion estudiante - banner"`, `"Radicar Solicitud Homologacion"`) into one,
`"Submit Request"`, for its own statistics. The mined `.pnml` still has two separate
transitions — nothing translated between the two, so tokens asking for either native label
found zero eligible actions in the (Module-1-derived, merged-only) action space. This affected
the *current, unmodified* production wiring, not just resource-pool experiments — confirmed by
building an unpatched `gym_env` for Consulta and finding both native labels present in
`self.task_types` but absent from every `self.output` pair. An empty mask on Consulta's very
first real activity meant Consulta could not be simulated in any configuration.

**Done, all in `gym_env.py`:**
- Load `self.activity_merge_map = artifact.get('config_used', {}).get('activity_merge_map') or {}`
  (already saved by Module 1 at merge time — no new data needed). Empty `{}` for every dataset
  that doesn't merge anything (BPI2017/BPI12W/Helpdesk) — confirmed no-op for those.
- Added `_canonical_activity(activity)` helper: `self.activity_merge_map.get(activity, activity)`.
- `self.task_types` now built by mapping every native `processing_time` key through
  `_canonical_activity` and deduplicating, instead of using native labels directly.
- Applied the same translation at every point a token's native `_next_activity` gets compared
  against the (merged) task-type/action space: `get_state()`'s pending-count feature,
  `step()`'s token-matching lookup, `check_possible_assignments()` (the function
  `next_decision_moment()` uses to decide whether to hand control back to the agent).
- A second, related crash found while testing the fix end-to-end: `process.py`'s
  `_resource_events` (a WIP-tracking `simpy.Resource`, capacity=`inf`, keyed by activity name)
  is built at `SimulationProcess.__init__` from the *native* `PROCESSING_TIME` keys, before
  `gym_env` has computed anything merged — `event_trace.py`'s
  `_get_resource_event(action['task'])` then looked up the merged name and got a `KeyError`.
  Fixed in `gym_env.reset()`, right after `SimulationProcess` construction: when
  `activity_merge_map` is non-empty, rebuild `simulation_process._resource_events` with every
  native key passed through `_canonical_activity` — both original activities now correctly
  share one WIP-tracking bucket, matching how they're treated everywhere else.
- Confirmed the Petri net's own transition/branching logic (`next_step()`,
  `define_xor_next_activity()`) operates purely on pm4py's marking/token-game state, never on
  activity-name strings, so this translation is safe — it doesn't touch process control flow.
  The only observable side effect is cosmetic: the trace-prefix/log output shows the merged name
  instead of the true original for these two activities.

**Validated:** Consulta now runs end-to-end — 181/181 traces complete cleanly, no crash, no
empty mask, both with the full resource pool and with a restricted (generalist-preferred, see
Phase 17) pool.

### Step 37 — `gym_env.py` / `train_DRL.py`: reproducibility — `reset(seed=...)` actually seeds

**Why:** Arrival timing (`inter_trigger_timer.py`), XOR-gateway branching
(`event_trace.py`'s `random.choices(...)`), and shock parameter sampling (`shock.py`) all draw
from Python's global `random` module and numpy's global `np.random` state. Nothing anywhere set
a seed. `gym_env.reset(seed=0, ...)` had a `seed` parameter (standard Gymnasium convention) but
never used it inside the method — pure dead parameter. Confirmed this was the cause of
run-to-run variance observed while calibrating `P*` candidates (two "identical" FIFO runs on
Consulta produced mean cycle times of 119,522s and 415,890s). `phi_fix`/`CPL_base`'s existing
calibration loops (Steps 28-30) had the same gap.

**Done:**
- `gym_env.py`: added `import random`. Changed `reset(self, seed=0, i=None)` to
  `reset(self, seed=None, i=None)`; at the top of the method, `if seed is not None:
  random.seed(seed); np.random.seed(seed)`. Default changed from `0` to `None` specifically so
  existing callers that don't pass a seed keep today's unseeded behaviour unchanged (training
  episodes should stay varied) — only callers that explicitly opt in via `reset(seed=k)` get
  reproducibility.
- `train_DRL.py`: `cal_env.reset()` → `cal_env.reset(seed=0)` (FIFO `CPL_base` calibration).
  `rand_env.reset()` → `rand_env.reset(seed=k)` inside the existing `for k in
  range(K_PHI_FIX_ROLLOUTS)` loop — reproducible as a whole K-rollout set while still sampling
  K genuinely different random-policy trajectories.

**Validated:** `reset(seed=42)` run twice produces byte-identical completed-trace cycle times;
`reset(seed=42)` vs `reset(seed=7)` diverge, as expected.

---

## Phase 17 — Validated methodology, not yet wired into production code

Two significant pieces of work were done and validated extensively via standalone scratchpad
scripts (monkey-patching a constructed `gym_env` instance) rather than by editing
`empirical_preprocessing.py`/`gym_env.py` directly — because both are still open design
decisions being iterated on, not yet ready to freeze into the actual artifact/environment code.
Recorded here so this document stays the authoritative "what's actually in the repo right now"
reference; neither of these is reflected in `RIMS_DRL-main`'s files as of Step 37.

**Resource-pool restriction (generalist-preferred, tier-based).** Discovered that
`gym_env.py`'s actual resource pool (`self.resources`, sourced from the legacy
`input_*20.json` file via `threshold=20`) is unrelated to Module 1's `n_min`-filtered
artifact — for BPI2017 this meant only 11 of 137 real resources were ever actually reachable,
several activities down to a single eligible resource, and at least one severe single-resource
bottleneck driven by a thin (n=6) support sample. Built and validated (BPI2017, BPI12W,
Consulta) a replacement pool: for each activity, the top-2-by-support resources in each of
Module 1's 3 tiers (slow/mid/fast), preferring candidates who are *also* reliably eligible for
other activities (a generalist-preference score) over narrow specialists — this keeps the
resulting pool's cross-activity flexibility close to the real dataset's (BPI2017: 50% vs. real
71%; Consulta: 52% vs. real 53%) instead of collapsing to near-fully-siloed specialists (an
earlier, simpler "rank by support alone" version gave only 17% cross-trained). Validated
end-to-end on all 3 simulatable datasets — healthy queues, no crashes, once a bug in the pool
builder itself (forgetting to carry over the placeholder-resource entries for
`activities_at_risk`) was found and fixed. **Not yet implemented:** wiring this pool
construction into `empirical_preprocessing.py`/`gym_env.py` for real — currently exists only as
a reusable scratchpad script (`build_generalist_pool.py`) plus a monkey-patching test harness.

**`P*` redefinition (simulator-native, seeded, median-based).** The historical p95 `P*`
(Step 23) is ~2,500-3,500x larger than anything the simulator's resource-contention dynamics
produce, even under the restricted pool above — confirmed not a bug (see
`Training_Readiness_Implementation_Summary.md` §8-9). Decided replacement methodology:
`P* = k × median(cycle times)` from a seeded, multi-rollout FIFO/no-shock calibration on the
*same* simulator and resource pool that will actually be trained on — analogous to how
`phi_fix`/`CPL_base` are already calibrated (Steps 28-30), rather than mixing a historical-log
statistic with simulator-native ones. Median (not mean, not a raw percentile) chosen after
direct empirical comparison: on Consulta's heavy-tailed distribution, the per-seed *mean*
varied 51% across 8 seeds (one seed's mean was 10x another's) versus the median's 9.5% —
mean and raw percentiles are both dominated by the same rare bottleneck-driven outliers a
percentile-of-history approach was already known to be vulnerable to. **Not yet implemented:**
these candidate `P*` values haven't been written into the artifact JSONs — `Module 1/artifacts/*.json`
still carry the original historical p95 value, and this recomputation only exists as a
scratchpad script (`p_star_seeded_calibration.py`) run manually per dataset.

---

## Phase 18 — Shock effectiveness: diagnosis and fixes

Testing the shock layer end-to-end for the first time with a resource pool small enough to
plausibly show contention (Phase 17's generalist pools, via scratchpad wiring) revealed that
neither shock type produced any measurable disruption — recovery was near-instant in every
paired shocked-vs-unshocked comparison, on all three simulatable datasets. This phase records
the diagnosis and the two resulting fixes.

### Diagnosis

Ruled out in order, each with direct evidence, not assumption:

1. **Not a measurement artifact.** Initially `ShockState.removed_resources` appeared empty after
   every `RESOURCE_REMOVAL` run — turned out to be reading it *after* the episode terminated,
   by which point `_apply_offset()` had already cleared it (correct cleanup). Snapshotting it
   immediately after `reset()` showed resources genuinely were being removed.
2. **Not a broken `RESOURCE_REMOVAL` mechanism.** Confirmed `RoleSimulator.removed` is correctly
   read in `process.py`'s `get_state()` (line 97) — removed resources really do become
   ineligible. Even removing the single most load-bearing resource in a pool (BPI12W's `10609`,
   ~40% of all assignments; Consulta's `1343`, a near-100%-utilized bottleneck) for up to 6 hours
   produced no visible backlog. Root cause: genuinely sparse arrivals — Consulta's real
   inter-arrival gaps have median 5,293s (~1.5h), mean 7,714s, and *zero* gaps under 10 minutes
   across 181 cases. A short-to-moderate resource outage often just doesn't overlap with enough
   simultaneous demand to matter. Not a bug — a property of the calibrated arrival process at
   these shock durations.
3. **`DEMAND_SPIKE` was completely non-functional — a real, confirmed bug**, and the primary
   explanation. `inter_trigger_timer.py`'s `get_next_arrival()` only applied
   `simulation_process.arrival_multiplier` inside the `'distribution'` arrival-generation branch.
   All four target datasets use `interTriggerTimer.type == 'custom'` (Prophet-forecast-based, a
   pre-generated arrival table) — that branch never read `arrival_multiplier` at all. Worse: even
   for `'distribution'`-type datasets it would have been a no-op regardless, because **every
   token's arrival is scheduled once, upfront, in `gym_env.reset()`'s token-creation loop — before
   warm-up, before any shock could possibly exist.** `arrival_multiplier` is always `1.0` at the
   moment `get_next_arrival()` runs. `DEMAND_SPIKE` had never once changed when a single case
   arrived, on any dataset, at any point in this project.

### Step 38 — `shock.py`: activity-scoped `RESOURCE_REMOVAL`

**Why:** with a small pool (~19-22 resources from Phase 17's generalist-pool design) and load
concentrated on 2-3 resources, a uniform-random draw from the *whole* pool usually misses what
matters — confirmed directly: a random 30% draw on BPI12W removed 5 resources, none of them the
two carrying ~75% of all work.

**Done:** `ShockConfig` gets a new `resource_removal_scope` field (`'activity'`, the new default,
or `'global'`, the old behaviour, kept for comparison) and `eval_target_activity` (eval mode
only, for controlled testing — random if unset). `ShockState` gets `removal_target_activity` for
introspection (cleared on `deactivate()`, same lifecycle as `removed_resources`). `_apply_onset`'s
`RESOURCE_REMOVAL` branch, when scope is `'activity'`, picks one activity (from
`simulation_process.resource_processing_times`, excluding placeholder-only activities) and draws
`magnitude` fraction from *that activity's own* eligible-resource pool (typically ~6 people from
the tier-based design) instead of the whole system's pool — the same magnitude is now a much
bigger fractional hit where it can actually create a queue. `resource_groups`, if configured,
still takes priority over either scope (unchanged).

### Step 39 — `process.py` / `event_trace.py` / `shock.py` / `inter_trigger_timer.py`: fix `DEMAND_SPIKE`

**Why:** see diagnosis point 3 above. Fixing this requires making a shock able to affect arrivals
that were already scheduled before the shock existed — not just changing a multiplier that
nothing was still reading.

**Done:**
- `process.py`: `SimulationProcess.__init__` gains `self.pending_arrival_procs = {}` — every
  not-yet-arrived token's `inter_trigger_time()` SimPy process, keyed by token id.
- `gym_env.py`: `reset()`'s token-creation loop now stores
  `self.simulation_process.pending_arrival_procs[i] = self.env.process(token.inter_trigger_time(itime))`
  instead of discarding the process handle.
- `event_trace.py`: `Token.inter_trigger_time()` rewritten to be interruptible. Tracks its
  remaining wait in "normal-pace" units (`remaining_normal`); each segment converts that to real
  sim-time by dividing by whichever `arrival_multiplier` is live *right now*, and catches
  `simpy.Interrupt` to recompute the remaining wait under a newly-changed multiplier instead of
  running to a fixed, pre-shock completion time. Also fixes a related correctness bug this
  surfaced: `_start_trace` (used for cycle-time accounting, `event_trace.py:196`) was set once at
  `Token.__init__` to the *planned* normal-pace arrival delay and never updated — under
  compression/decompression a token can arrive earlier or later than that plan, so cycle times
  computed from the stale value would have been wrong. Now set to `self.env.now` at the true
  arrival instant, once the wait actually completes.
- `shock.py`: added `_interrupt_pending_arrivals(simulation_process)` — interrupts every process
  in `pending_arrival_procs` (catching `RuntimeError` for any that already finished/arrived, and
  pruning them from the dict). Called from `_apply_onset`'s `DEMAND_SPIKE` branch (after setting
  the new multiplier) and symmetrically from `_apply_offset`'s (after resetting it to `1.0`), so
  still-waiting tokens both compress at onset and decompress back to normal pace at offset.
- `inter_trigger_timer.py`: removed the old `arrival = arrival / self._process.arrival_multiplier`
  line from the `'distribution'` branch — confirmed always a no-op (see diagnosis point 3) and
  superseded by the uniform, dataset-type-agnostic interrupt mechanism above.

**Validated:**
- Direct arrival-timestamp instrumentation (not just downstream effects): a `magnitude=5` (6x
  rate), 6-hour `DEMAND_SPIKE` on Consulta produced 36 arrivals during the shock window (median
  gap 401s) versus 1 arrival in an equal-length window immediately after, once normal pace
  resumed — confirmed compression is real and working.
- End-to-end paired shocked-vs-unshocked comparisons, same seed, generalist pools + Phase 17's
  new `P*`:

  | Dataset | Baseline max `n_pending` | Shocked max `n_pending` | Recovery after shock ended |
  |---|---|---|---|
  | BPI2017 | 2 | 6 | 9.06h |
  | Consulta | 2 | 15 (sustained) | 0.33h |
  | BPI12W | 1 | 2 | 0.02h |

  First genuinely real, measurable disruption-and-recovery signal produced in this project —
  every prior shock test (Phases 15-17, and the start of this phase) showed near-instant
  "recovery" because there was never a real disruption to recover from. The relative ordering
  (Consulta > BPI2017 > BPI12W) tracks each dataset's baseline arrival sparsity, which is an
  expected/sane pattern, not noise.

---

## Phase 19 — Real data-quality bug found and fixed: Consulta's `aggregate()` fallback

Attempting to compute the four standard resilience metrics (Time to Impact, Maximal Performance
Deviation, Recovery Time, Cumulative Performance Loss — definitions in
`Training_Readiness_Implementation_Summary.md` §27) on Consulta surfaced a multi-million-second
`phi(t)` blowup in *every* seed, shocked or not. Traced to a real bug in Module 1, not a
simulator or shock issue.

### Step 40 — `Module 1/empirical_preprocessing.py`: remove `aggregate()`'s fallback re-admission

**Why:** `aggregate()`'s fallback (see original Step-era docstring) re-admitted the
highest-support candidate for any activity that would otherwise have zero eligible resources
after the `n_min` filter — intended to prevent simulator deadlock on rare activities. For
Consulta's `"Transferir Creditos"`, every candidate failed `n_min=3`; the fallback re-admitted
resource `'1343'` anyway, with support=2 and a median processing time of 1,435,767s (16.6 days)
computed from just those 2 raw observations. Because it became the *sole, always-selected*
resource for that activity, every case reaching it was forced through a 16.6-day wait —
single-handedly dominating simulated cycle times whenever a case happened to hit it, which is
exactly what a full-population run eventually does. Traced by instrumenting a live run and
finding the specific `(resource, activity)` pair via `support_counts`/`resource_processing_times`
— not guessed. A second, harmless instance of the same bug also existed (`'19019'` on
`"Cancelar curso"`, support=2, but a benign 1,519s median).

**Done:** `aggregate()`'s fallback block removed entirely. An activity where every candidate
fails `n_min` is now left with zero eligible resources on purpose — `validate()` flags it as
at-risk, and it gets the existing synthetic placeholder resource (dataset-median duration,
`math.inf` capacity — Step 22/27) instead of a re-admitted 1-2-observation outlier. Docstring
updated to explain why re-admission was removed, referencing this specific incident.

**Validated:** regenerated `Module 1/artifacts/consulta.json` (same CLI invocation/config as
before: `n_min=3`, same `activity_merge_map`/`exclude_activities`) — `activities_at_risk` grew
from 4 to 6 (`"Transferir Creditos"` and `"Cancelar curso"` now correctly at-risk), and the
`excluded_pairs` list no longer contains any `_FALLBACK_READMITTED` entries. Re-ran the
generalist-pool builder and an 8-seed unshocked baseline: pooled `phi(t)` max dropped from
3,743,164s to 27,873.7s — the runaway is gone. Backed up the pre-fix artifact as
`consulta.json.bak_pre_fallback_fix` before regenerating.

---

## Phase 20 — Validated methodology from resilience-metric testing, not yet wired into production code

Extensive work computing and stress-testing the four resilience metrics (TTI, MPD, Recovery
Time, CPL — see `Training_Readiness_Implementation_Summary.md` §27-35 for the full narrative)
surfaced several more findings and a working pool-construction technique. As with Phase 17, all
of this was validated via scratchpad scripts that build a `gym_env` instance and monkey-patch its
resource pool/policy, not by editing `gym_env.py`/`train_DRL.py`/`process.py` directly — these
remain open decisions, not yet frozen into the production files.

**Baseline resource-selection policy: `valid[0]` ("FIFO") silently concentrates load.**
`retrieve_possible_assignments()` (Step 7) returns actions `sorted()` by `(resource, activity)`
tuple — i.e., alphabetically/lexicographically by resource name. Every calibration and shock test
in this project through Phase 18 used `action = valid[0]` as its "FIFO" baseline policy, which
therefore always tries the *alphabetically-first eligible resource* for whatever activity is
pending, not a load-balanced choice. Confirmed via assignment-count instrumentation: on
Consulta's generalist pool, the top resource absorbed 32.3% of all real assignments under
`valid[0]`, vs. 7.8% under a simple least-used-resource policy (pick the valid action whose
resource has the fewest cumulative assignments so far). Switching to least-used is a real
methodological correctness fix (a policy that never touches ~90% of the available pool is not a
meaningful resilience-testing or calibration baseline) and was adopted for every result in
Phase 20/21 onward — but it does **not** by itself resolve pool fragility: re-testing Consulta's
demand-spike grid under least-used showed the same catastrophic/no-effect split as `valid[0]`,
just with different seeds landing on each side. **Not yet implemented:** `train_DRL.py`'s FIFO
calibration loops (Steps 28, 30) still use `valid[0]`.

**`max_recovery_wait_s` backstop was miscalibrated, not a sign of genuine non-recovery.** The
single-shock-per-episode design's backstop (Step 35, `gym_env.py:96-97`) is
`5 × shock_config.duration_range[1]` — for a 20-hour shock, a 100-hour window. Direct test: a
Consulta run reported as "TRUNCATED — did not recover" at that backstop, re-run with
`max_recovery_wait_s` set to infinite, genuinely recovered at 175.2 hours (7.3 days). Every
earlier "catastrophic, never recovers" finding in this project (including some of Phase 18/19's
own shock-comparison results) needs re-reading in this light — some of it was real non-recovery,
some was this backstop firing too early. **Not yet implemented:** the `5×duration` formula is
unchanged in `gym_env.py`; the fix applied throughout Phase 20/21's testing was disabling the
backstop (`env.max_recovery_wait_s = float('inf')`) for evaluation runs specifically, since
population exhaustion already guarantees episode termination for these finite-trace datasets —
a real fix would likely gate the backstop to `train` mode only, or derive it from observed
recovery times rather than shock duration.

**`N_TRACES` population exhaustion was an easy, already-available fix, not a real ceiling.**
Several "catastrophic, never recovers" results turned out to be the case population running out
before recovery could be observed (episode terminates via `len(self.tokens) == 0`, not via
`phi(t) <= P*`) — not a backstop issue, a genuinely finite dataset issue (Consulta: 181 real
cases; BPI12W: 1,200). Both datasets' `interTriggerTimer.type == 'custom'` arrivals are generated
by `InterTriggerTimer.generate_arrivals()` (`inter_trigger_timer.py:69-116`) via a Prophet model
that already extrapolates arrivals arbitrarily far into the future in 30-day chunks until it has
enough for whatever `num_instances` (`N_TRACES`) is requested — it is **not** a fixed replay of
the historical case count. Passing a larger `N_TRACES` (e.g. Consulta 181→400, BPI12W tested up
to 4,000) works with zero code changes and lets the *same* fitted arrival-rate model keep
generating statistically consistent future arrivals, which is more principled than manually
looping the historical log. Validated: a Consulta seed that hit "population exhausted, never
recovered" at `N_TRACES=181` genuinely recovered (RT=255.7h) once given `N_TRACES=400`, in 0.8s of
wall time. **Not yet implemented:** `gym_env.py`'s `N_TRACES == 'from_input_data'` default is
unchanged; all extended-population testing passed a literal larger integer instead.

**Zero-specialist ("pure generalist") pool construction — a new algorithm, not yet in
`empirical_preprocessing.py`.** The existing generalist-preferred pool builder (Phase 17,
`build_generalist_pool.py`) only *prefers* cross-trained candidates as a tie-break within each
`(activity, tier)` slot chosen independently — it does not guarantee every selected resource ends
up used in 2+ activities. Built a proper constraint-satisfaction variant
(`build_pure_generalist_pool_v2.py`, scratchpad): a backtracking search over all
`(activity, tier)` slots, restricted to candidates with `degree(r) >= 2` (eligible for 2+ distinct
tierable activities in the artifact's own `resource_tier_map`), with forward-checking (every
partially-placed resource must still have a reachable open slot, checked after *every*
assignment, not just at the leaf) to avoid the exponential blowup a naive greedy-then-patch
approach hit. Two earlier, simpler repair strategies (greedy + single-hop swap; greedy +
bounded-depth augmenting-path chain) both got stuck in a whack-a-mole pattern — fixing one
stranded resource created a new one elsewhere — before the full backtracking version resolved it
cleanly. Feasibility varies by dataset: BPI2017 has a fully feasible zero-specialist assignment
(9 real resources, pool shrunk from 19 in the original 2-per-tier design); BPI12W does not — its
fraud-assessment activity's mid and fast tiers each have exactly one historical
resource, neither cross-trained in the whole log — accepted as two explicit, documented
exceptions (a genuine data constraint, not an algorithm failure) rather than silently violated or
worked around by misrepresenting tier speed. **Not yet implemented:** this construction only
exists as the scratchpad script; not wired into `empirical_preprocessing.py`.

**`P*` methodology: locked to a fixed `2 × pooled median(cycle_times)`, not a baseline
percentile.** Phase 17's `k × median` idea was tried in a baseline-φ(t)-percentile form during
this testing (P* = 90th percentile of unshocked φ(t) samples) specifically to get a
statistically-grounded, noise-aware threshold — but this backfired on the zero-specialist BPI2017
pool: forcing full cross-training made *baseline* scheduling noise measurably worse (unshocked
φ(t) p90 rose from 3,263s at 2-per-tier to 10,236s at zero-specialist), and since P* was
recalibrated per-pool as that same noisy baseline's 90th percentile, the threshold rose right
along with it — the exact same shocks that clearly should have shown an effect reported "no
breach at all" purely because the yardstick moved, not because the pool got more robust. Fixed by
locking `P* = 2 × pooled_median(raw completed-trace cycle times)` from a seeded, least-used-policy,
no-shock baseline (same computation as Phase 17's original candidate, `k=2` rather than `k=3`) —
computed once per pool and held fixed across every subsequent magnitude/duration test on that
pool, not recalibrated per test. Re-running BPI2017's zero-specialist pool under this fixed value
(P*=3,803.3s) recovered a strong, clean result (15 of 16 seed×magnitude combinations breach,
recovery 0-32.8h) — confirming the earlier "no breach anywhere" was entirely an artifact of the
percentile-based method, not the pool. **Not yet implemented:** artifacts still carry the
original historical-p95 `P_star`; this `2×median` value is computed ad hoc per pool in scratchpad
scripts, same integration gap as Phase 17.

**Realistic shock-magnitude ceiling, checked directly against the event logs.** `DEMAND_SPIKE`'s
`magnitude` (`arrival_multiplier = 1 + magnitude`, Step 39) has no connection to the data — its
own default `magnitude_range = (0.2, 0.5)` (a 20-50% surge) was checked against each log's actual
historical daily arrival-count volatility: Consulta's worst single day ever recorded was 3.90x
its mean daily volume (p95/median 3.32x), BPI12W 2.72x (p95/median 1.98x), BPI2017 2.07x
(p95/median 1.53x) — none of the three logs ever saw a *sustained* surge above ~4x, over any
window comparable to the 6-20 hour shock durations used in testing. Magnitude 5-10 (a sustained
6x-11x rate increase) is therefore already somewhat beyond anything historically observed in any
of these three logs — defensible as a "severe, unprecedented event" scenario, but not something
drawn from the data — while the magnitude 20-100 range used earlier in Phase 18/19's exploration
has no connection to plausible reality at all and was purely a diagnostic stress-test range, not
a claim about realistic demand.

**Cross-dataset resilience-profile comparison, same pool-construction method + fixed P\* for
all.** With the above fixes in place, three genuinely comparable dose-response results (duration
21600s/6h except where noted; magnitude 5-8 unless noted):

| Dataset | Pool | P\* (2×median) | Realistic-magnitude result |
|---|---|---|---|
| Consulta | 23-resource generalist (unmodified, Phase 17 design) | 37,063.5s | Strong — 16/16 combos recover cleanly once `N_TRACES` extended to 400 and backstop disabled (duration 20h); TTI flat ~11-15h, MPD/CPL smooth in magnitude, RT 125-256h |
| BPI2017 | 9-resource zero-specialist (Phase 20) | 3,803.3s | Strong — 15/16 combos breach; MPD 1.14x-2.44x, RT 0-32.8h |
| BPI12W | 9-resource zero-specialist (Phase 20, 2 documented exceptions) | 1,558.5s | Weak — only 5/16 combos breach, all marginal (MPD 1.01x-1.36x), recovering in <=2.2h every time |

BPI12W's robustness under this identical, controlled methodology (same construction algorithm,
same fixed P* definition, same lean/fully-cross-trained staffing as the other two) is now a real
cross-dataset finding, not an artifact of a weaker pool or a differently-calibrated threshold —
BPI2012's process genuinely has more slack relative to its own demand than either BPI2017 or
Consulta.

---

## Phase 21 — Fixed calibration (P*, cpl_base, phi_fix) wired into production for real

Phase 20's methodology (least-used policy, no backstop, curated pools, fixed `2×median` `P*`)
was validated entirely via scratchpad monkey-patching. Before training, the user required these
to be real, checked-in, persisted artifacts — never recomputed live inside `train_DRL.py` — so
every simulation on a given (dataset, pool) pair uses identical numbers. This phase does that.

### Step 41 — `gym_env.py`: remove the recovery backstop entirely

**Why:** `max_recovery_wait_s` (`5 × shock_duration`) was proven in Phase 20 to truncate episodes
before real recovery happened. The user's instruction: no episode anywhere — calibration,
training, or eval — should be time-boxed. Recovery happens whenever it happens, or the episode
ends when the trace population runs out.

**Done:** deleted `self.max_recovery_wait_s` (`__init__`) and the `isTruncated`-via-backstop
branch in `step()`. `episode_done = len(self.tokens) == 0 or recovered` is now the sole
termination condition; `step()` always returns `isTruncated=False` (kept in the tuple only for
Gymnasium API compatibility).

### Step 42 — `gym_env.py`: `pool_override_path` and `p_star_override` constructor params

**Why:** every scratchpad script this session duplicated ~15 lines to monkey-patch `resources`,
`resource_processing_times`, `output`, `observation_space`/`action_space`, and
`params.ROLE_CAPACITY` from a pool JSON. Folding this into `gym_env.__init__` itself means the
calibration script and `train_DRL.py` share one real code path instead of each re-implementing
the monkey-patch. This is *not* the full "wire the pool into Module 1" task — `empirical_preprocessing.py`
is untouched, and no pool is made a default anywhere; it only takes effect when explicitly passed.

**Done:** `pool_override_path`, if given, loads `{"curated_pt": ..., "resource_to_activities": ...}`
right after the artifact-driven resource list is built, replacing `resource_processing_times` and
`resources` with the curated pool's; `ROLE_CAPACITY` is rebuilt the same way immediately after
`self.params` is constructed (calendars sourced from the same input file, default business-hours
calendar as fallback). `p_star_override`, if given, takes priority over the artifact's own
`P_star` (still the historical p95) wherever `self.P_star` is set.

### Step 43 — new file `calibrate_resilience_constants.py`

**Why:** "if we do not have a concrete script... create the necessary scripts" — a real, checked-in
replacement for every scratchpad P*/CPL_base/phi_fix calculation this session.

**Done:** computes and persists, for one (dataset, pool) pair:
- `P_star = 2 × pooled_median(cycle_times)` from `n_baseline_seeds` seeded, no-shock, least-used-
  policy episodes, each run to full completion (no wall-clock cap, no backstop — `--n-traces`
  must be chosen generously; the script does not compensate for too small a value).
- `cpl_base` — **mean of `cpl_accumulated` pooled across `n_baseline_seeds` seeded episodes**
  under one fixed, representative shock (not a single seed=0 episode, matching the original FIFO
  calibration's convention — a single-episode design was tried first and found to risk landing on
  a non-breaching seed, silently giving `cpl_base=0.0` and disabling RF2's entire
  constraint-violation penalty for the whole training run; confirmed happening on Consulta before
  this was fixed. See the Consulta results below.).
- `phi_fix` — worst `phi(t)` across `n_phi_fix_rollouts` random-policy episodes under the same
  fixed shock (unchanged design from the original Step 30, just now checked-in and reusable per
  pool).
- Writes one JSON (`{"P_star", "cpl_base", "phi_fix", "config_used": {...full provenance...}}`).

Reuses the `least_used_action()` pattern from this session's `generic_baseline_and_shock.py`/
`shock_grid_only.py` scratchpad scripts, promoted into checked-in code.

### Step 44 — `train_DRL.py`: load the calibration file instead of recomputing inline

**Why:** the same "never recomputed live" requirement — a fresh FIFO/random-policy calibration on
every launch risks a different number each time (different pool, different random draw), and
there is no way to guarantee two training runs used the same `P*`/`cpl_base`/`phi_fix` unless
that number is read from one shared, persisted source.

**Done:** removed the inline FIFO `cal_env` block and random-policy `phi_fix` block. New CLI
positional args (8th, 9th): `pool_override_path`, `calibration_path`. If `shock_config is not
None`, `calibration_path` is required — raises immediately (not a silent zero/`None` fallback) if
missing, or if the calibration file's own recorded `dataset`/`pool` don't match what's being
requested. `p_star_override`/`pool_override_path` are threaded through to both `env_simulator` and
`eval_env` construction, so training and eval share the exact pool and `P*` the calibration file
was computed from.

### Calibration results so far

**Consulta** (`consulta_generalist_pool.json`, 23-resource generalist pool, `n_traces=2000`,
`n_baseline_seeds=8`, shock=`DEMAND_SPIKE` magnitude=6/duration=72000s):

| | Value |
|---|---|
| `P_star` | 32,856.0s |
| `cpl_base` | 33,710,522,992.91 (mean across 8 seeds — 3 never breached `P*` at all, cpl_accumulated=0; the other 5 ranged 44.2-71.5 billion) |
| `phi_fix` | 236,243.55s (dominated by one outlier rollout out of 5 — the others clustered 26,000-31,500s) |

Written to `Module 1/artifacts/consulta_calibration.json`.

**BPI2017: initially paused, later completed — see Step 45.** The `P*` baseline alone (8 seeded
episodes, each run to full completion of all 5,789 traces, no wall-clock cap per the user's
instruction) took roughly 23 minutes for a *single* seed — projecting to several hours for `P*`
alone, before `cpl_base`/`phi_fix` even start. Stopped deliberately due to computation-budget
constraints at the time; Consulta was prioritized to completion first since its smaller trace
count made it tractable in that session. Resumed and completed in a later session — see Step 45
for the fix and BPI2017's final results.

### Step 45 — `calibrate_resilience_constants.py`: `--n-traces-baseline`, and BPI2017 finished

**Why:** the ~23-minutes-per-seed cost above was assumed to belong to the whole calibration run.
It doesn't. `gym_env.step()`'s recovery-based termination (`episode_done = len(tokens) == 0 or
recovered`) only fires when a shock was actually fired that episode (`recovered` requires
`self.shock_fired is not None`). `cpl_base`/`phi_fix` both run under the fixed shock, so those
episodes were already terminating on recovery, well short of `n_traces` — confirmed in the
BPI2017 rerun below, where shocked episodes completed after roughly 90-130 traces each, not 5,789.
`P*` runs with `shock_config=None` by construction (it's the shock-free baseline), so `recovered`
can never be `True` there — every `P*` seed was forced to exhaust its full trace population. The
23-minute cost belonged entirely to that one stage.

**Done:** added `--n-traces-baseline` (optional, defaults to `--n-traces` if omitted), read only
by `calibrate_p_star()`; `build_env()` gained a matching `n_traces` parameter so callers can pass
a stage-specific trace count instead of always using `args.n_traces`. `--n-traces` continues to
size `cpl_base`/`phi_fix`'s episodes (a harmless ceiling there, since they terminate on recovery
well before it). `config_used` in the output JSON now records both `n_traces` and
`n_traces_baseline`. No change to the calibration formulas themselves — same `2×pooled median`,
same mean-pooled `cpl_base`, same max-of-`K`-rollouts `phi_fix`.

**BPI2017's final calibration** (`bpi2017_pure_generalist_pool.json`, 9-resource zero-specialist
pool, `n_traces_baseline=2000` for `P*`, `n_traces=5789` ceiling for `cpl_base`/`phi_fix`,
`n_baseline_seeds=8`, shock=`DEMAND_SPIKE` magnitude=6/duration=21600s):

| | Value |
|---|---|
| `P_star` | 3,816.5s |
| `cpl_base` | 167,982,158.72 (mean across 8 seeds — all 8 breached `P*`, unlike Consulta's 3-of-8 zero-breach case) |
| `phi_fix` | 8,092.11s |

Written to `Module 1/artifacts/bpi2017_calibration.json`. **BPI2017 is calibration-complete and
training-ready**, on the same footing as Consulta.

### Step 46 — `train_DRL.py`: train-time `shock_config` aligned to the calibrated/eval shock

**Why:** train-mode `shock_config` had never been touched since the original codebase defaults —
`magnitude_range=(0.2, 0.5)` (a 1.2x-1.5x arrival bump) and `duration_range=(1800.0, 7200.0)`
(30min-2h). Every calibration and resilience-metric run this project has produced
(`calibrate_resilience_constants.py`, `compute_resilience_metrics.py`, Step 45) uses magnitude in
`[5, 8]` (a 6x-9x arrival multiplier — `DEMAND_SPIKE`'s multiplier is `1 + magnitude`) and a fixed
21,600s (6h) duration — the validated realistic range from
`Training_Readiness_Implementation_Summary.md` §32. Training on the old mild range while being
evaluated against the severe one would mean the agent rarely, if ever, practices on a shock strong
enough to need a genuine recovery response, then gets judged on exactly that. Flagged as the one
remaining concrete gap before either dataset was truly training-ready (§38/§40 of the Summary
doc), and fixed once asked.

**Done:** `magnitude_range` changed `(0.2, 0.5)` → `(5.0, 8.0)`; `duration_range` changed
`(1800.0, 7200.0)` → `(21600.0, 21600.0)` (a degenerate range — `random.uniform(21600.0, 21600.0)`
always returns `21600.0`, so this is effectively a fixed duration, matching calibration/eval
exactly rather than sampling around it). `shock_types` (`['RESOURCE_REMOVAL', 'DEMAND_SPIKE']`)
left unchanged — untouched, not evaluated as part of this fix.

**Verified:** `ast.parse` on the edited file; `random.uniform` on a degenerate range confirmed to
return the fixed value, not raise.

### Step 47 — new `Modules 2-3/Helpdesk Preprocessing/build_helpdesk_engine_bundle.py` + `verify_helpdesk_bundle.py`

**Why:** Unlike BPI2017/Consulta/PurchasingExample, Helpdesk shipped with no pre-existing
`RIMS_DRL-main/example/<NAME_LOG>/` bundle at all — no `.pnml`, no `input_Helpdesk.json`. Module 1
being complete for a dataset does not make it simulatable; the base engine's own per-dataset
bundle has to exist too.

**Done:** `build_helpdesk_engine_bundle.py` discovers a sound WF-net Petri net via
`pm4py.discover_petri_net_inductive` on the raw CSV (all 9 activities, unfiltered — Module 1's own
`config_used` confirms no `W_`-prefix/lifecycle filtering applies here), and assembles
`input_Helpdesk.json` (`interTriggerTimer.type="distribution"` fit to real inter-arrival gaps,
rather than `"custom"`/Prophet, since no trained arrival model exists for this log; `resource`
dict excluding the `__NO_RESOURCE__` placeholder, which `gym_env.py` builds programmatically
regardless of the input file; `resource_table`/`probability` either inert placeholders or omitted
entirely, both confirmed dead/defaulted paths). A minimal `Helpdesk_dispr_meta.json` is also
generated — required only because `Parameters.parameters_LSTM_model()` crashes without a METADATA
file present (`PR_WIP_INITIAL` never set), even though every value it provides is itself dead code
(traced every read; the one live call site is gated behind `FEATURE_ROLE=='all_role'`, which this
file avoids by using the `_dispr` prefix). Bundle written to both the review folder and
`RIMS_DRL-main/example/Helpdesk/`, since `gym_env.py`'s lookup path is hardcoded to the latter.
`verify_helpdesk_bundle.py` round-trips the `.pnml` and runs a short `gym_env` construction +
episode to confirm the bundle actually works before trusting it.

**Found via this step, fixed in Step 48:** running the verification script surfaced a real crash
in the base engine, not a bundle-construction mistake — see Step 48.

### Step 48 — `inter_trigger_timer.py`: fixed a genuinely broken arrival mode

**Why:** `verify_helpdesk_bundle.py` (Step 47) hung/deadlocked — tracing it down (event-queue
instrumentation, `Token.simulation()` tracing) showed every one of Helpdesk's traces arriving at
`env.now=0` simultaneously, flooding `next_decision_moment()` (which drains one event per call)
before any real assignment could happen. Root cause: `get_next_arrival()`'s `'distribution'`
branch computed `self._interval` from the drawn value but never folded it into `self._previous`
(the line that would, `self._previous += self._interval`, was commented out) — `self._previous`
stays at its `__init__` default of `0` forever, so every arrival's computed delay is discarded.
This is not something the `'custom'` (Prophet-based) path used by BPI2017/Consulta could ever have
hit, since it sets `self._previous` directly on its own line — the `'distribution'` mode had
apparently never been exercised end-to-end before this session.

**Done:** Restored `self._previous += self._interval`, scoped to the `'distribution'` branch only
(the `'custom'` branch's own direct assignment is untouched, so this cannot change BPI2017/
Consulta's already-validated behaviour).

**Verified:** re-ran `verify_helpdesk_bundle.py` — a 100-trace no-shock episode now completes
cleanly (100/100 traces, mean cycle time ≈48 days, consistent with the log's own ≈39-day median).

### Step 49 — new `Module 1/concurrency_oracle_estimator.py` + `Module 1/validate_concurrency_oracle.py`

**Why:** Helpdesk has only single (completion) timestamps per event, no start/complete pairs, so
Module 1's existing `_compute_sojourn_pt` (gap since the previous event in the case) is the only
duration source available for it — and it turned out to badly overshoot reality (Step 50). Before
building a replacement, the replacement itself needed validating against known ground truth.

**Done:** `concurrency_oracle_estimator.py` implements per-event start-time estimation via a
Directly-Follows concurrency oracle (two activities are "concurrent" if the log shows both
directly-follows orderings between them) plus `max(case_enabling_time, resource_available_time)`
— the general technique from Chapela-Campa/Dumas et al.'s start-time-estimation line of work,
reimplemented directly rather than depending on the archived `start-time-estimator` package (its
successor, `pix-framework`, requires Python <3.12, incompatible with this environment).
`validate_concurrency_oracle.py` hides BPI2017's and Consulta's real start timestamps and checks
how well both the naive sojourn approximation and the new estimator recover the known-true
per-(resource, activity) durations. Result: the concurrency-oracle estimator is a real
improvement (~36x lower median absolute error than naive on BPI2017), but both methods perform
poorly on the "human-scale" (non-instant) activities that actually drive simulated congestion —
near-zero/negative correlation with ground truth on BPI2017 specifically, better (positive,
~0.32) on Consulta. Read as: per-event duration recovery from single-timestamp logs remains
unreliable for processes with heavy resource multitasking (BPI2017), less so for more
sequential/one-case-at-a-time processes (Consulta).

### Step 50 — new `Modules 2-3/Helpdesk Preprocessing/build_helpdesk_co_pool.py` + `calibrate_helpdesk_duration_scale.py`

**Why:** Applying the naive sojourn approximation to Helpdesk's own pool inflated its simulated
`P*` to ~1,130 days (29x the real log's own median cycle time) — resources were modeled as fully
blocked for the entire sojourn gap, most of which is real-world non-blocking wait time. Applying
the (BPI2017/Consulta-validated) concurrency-oracle estimator instead *undershot* by ~71x (0.5
simulated days vs. 39.1 real days) — Helpdesk's ~21 resources each handle hundreds to thousands of
tickets over the log's span, so "when did this resource last do anything" collapses to "almost
always very recently," stripping out genuine wait time along with the artificial kind. Neither
per-event method transfers to this dataset.

**Done:** `build_helpdesk_co_pool.py` builds a CO-based pool variant for reference (kept, not
used further). `calibrate_helpdesk_duration_scale.py` instead calibrates a single dataset-wide
scale factor `k`, applied uniformly to the *existing naive-pool durations'* relative structure,
found via bisection search in log-space against the one statistic reliably known for this
dataset: the real log's own median cycle time (3,375,171s/39.1d). This follows the same
methodological pattern as Simod's (Camargo, Dumas & González-Rojas, 2020, *Decision Support
Systems*) hyper-parameter search against a real-log-derived similarity measure, and general DES
model-calibration practice (De Santis et al., 2021, *Annals of Operations Research*; Law &
Kelton, *Simulation Modeling and Analysis*) — search over a free parameter until simulated output
matches a real output statistic, then validate against an independent one not used in the fit.
Full citation text: `Modules 2-3/Helpdesk Preprocessing/Duration_Calibration_Methodology.md`.

**Result:** converged at `k=0.36478` (5 iterations). Fit target (median): +14.8% (within the 15%
tolerance). Held-out checks: mean +8.2% (better than the fit itself), p95 +36.8% (a documented
limitation — a single uniform scale factor corrects overall pace but not tail/variance shape).

### Step 51 — `calibrate_resilience_constants.py`: `--p-star-formula {median,mean}` option

**Why:** PurchasingExample's cycle-time distribution turned out to be sharply bimodal (pooled
median 0.39d vs. pooled mean 8.45d across 8 no-shock seeds) — `P* = 2×median`, the standard
definition used everywhere else, left the baseline (no-shock) system in breach 85-93% of the
time, purely from normal variance in a rolling-mean statistic being compared against a threshold
anchored to the fast majority of a bimodal distribution.

**Done:** Added `--p-star-formula` (`'median'` default, unaffected for every existing dataset;
`'mean'` as an explicit, dataset-scoped override). `calibrate_p_star()` now also always prints the
pooled mean alongside the median, regardless of which formula is selected, so the choice is
auditable from the log. Recorded in `config_used.p_star_formula`. Verified only via PurchasingExample
so far — `2×mean` reduced baseline breach to 26-37% (better, though not fully resolved) — see
`Training_Readiness_Implementation_Summary.md` §47.

### Step 52 — `calibrate_resilience_constants.py`: `--pool` made optional

**Why:** PurchasingExample's resource distribution (27 named individuals, minimum 3 activities per
resource) didn't need a curated "generalist" pool the way BPI2017/Consulta/Helpdesk's did — the
full artifact-driven resource set was already healthy. The CLI's `--pool` was `required=True`,
forcing every dataset through a pool file even when there was nothing to curate.

**Done:** Changed to `default=None`. `build_env`/`gym_env` already handled `pool_override_path=None`
correctly (falls back to the artifact's own resource set) — this was a CLI-parser-only fix, no
behavioural change to any dataset that does pass `--pool`.

### Step 53 — `train_DRL.py`: removed a dead, version-incompatible import

**Why:** A Consulta training smoke test (first attempt) crashed immediately on startup:
`ImportError: cannot import name 'normalize' from 'gymnasium.wrappers'`. The installed gymnasium
version (1.2.3) restructured that module; `normalize` is no longer exported there.

**Done:** Grepped the file — `normalize` was imported and never referenced anywhere else.
Removed the dead import line entirely rather than chasing a currently-correct replacement import
for code that isn't used.

**Verified:** re-ran the smoke test; import error gone, training proceeded to environment
construction.

### Step 54 — `train_DRL.py`: train-time `shock_types` restricted to `['DEMAND_SPIKE']`

**Why:** With the Step 53 fix applied, the same smoke test ran real training rollouts
successfully for a while, then crashed with `IndexError: list index out of range` at
`gym_env.py:450` (`token_id = list(tokens_pending.keys())[0]`), immediately after a
`RESOURCE_REMOVAL` shock fired with `magnitude=7.753`. Root cause: `RESOURCE_REMOVAL`'s
`magnitude` means "fraction of the eligible pool to remove" (`n_remove =
max(1, math.floor(len(pool) * magnitude))`), but `train_DRL.py`'s `shock_config` shares
`magnitude_range=(5.0, 8.0)` with `DEMAND_SPIKE`, where the same range means a 6x-9x arrival
multiplier (`1+magnitude`) — a magnitude of 5-8 applied as a *removal fraction* clamps to removing
100% of an activity's eligible resources. A token already pending for that activity is then
stranded with zero eligible resources for the rest of the episode — a case Module 1's
`activities_at_risk`/placeholder-resource mechanism was never designed to cover, since that's a
static, historical-data-driven check, not a runtime one. The exact mechanism connecting the
stranded token to *this specific* `IndexError` a few steps later wasn't fully traced before this
was deferred (per explicit user instruction: shock magnitude/duration is not to be adjusted
without asking first — see the standing-rule note in `Training_Readiness_Implementation_Summary.md`
§45).

**Done (interim fix, not a real fix):** `shock_types` narrowed from
`['RESOURCE_REMOVAL', 'DEMAND_SPIKE']` to `['DEMAND_SPIKE']` only, with an inline comment
explaining why and what needs to happen before re-enabling `RESOURCE_REMOVAL` (its own,
correctly-scaled `[0,1]`-range magnitude, separate from `DEMAND_SPIKE`'s).

**Verified:** re-ran the smoke test (Consulta, 5,120 steps) — completed cleanly, only
`DEMAND_SPIKE` shocks fired, one full PPO rollout + policy update completed, `model_final.zip`
saved. First successful end-to-end training smoke test this project has produced.

### Step 55 — `shock.py`, `train_DRL.py`: `RESOURCE_REMOVAL` given its own magnitude range (real fix, superseding Step 54's interim `DEMAND_SPIKE`-only restriction)

**Why:** Step 54's restriction avoided the crash but didn't fix the actual bug — `RESOURCE_REMOVAL`
and `DEMAND_SPIKE` were sharing one `ShockConfig.magnitude_range` field with incompatible
semantics (a multiplier for one, a removal fraction that must stay in `[0,1]` for the other).

**Done:** `ShockConfig` gained a new field, `resource_removal_magnitude_range` (default `(0.2,
0.5)`, this project's original, never-recalibrated `RESOURCE_REMOVAL` range — unrelated to
`DEMAND_SPIKE`'s own calibration-driven `(5.0, 8.0)`). Both places that draw a train-mode
magnitude (`shock_scheduler`'s and `fire_single_shock`'s `else` branches) now select
`resource_removal_magnitude_range` when `shock_type == 'RESOURCE_REMOVAL'`, `magnitude_range`
otherwise. `magnitude_range`'s docstring updated to state it is now `DEMAND_SPIKE`-only.
`train_DRL.py`'s `shock_config` re-enables `shock_types=['RESOURCE_REMOVAL', 'DEMAND_SPIKE']` and
sets `resource_removal_magnitude_range=(0.2, 0.5)` explicitly.

**Verified:** re-ran the smoke test (Consulta, 5,120 steps, RF1) with both shock types active.
213 shocks fired (101 `DEMAND_SPIKE`, 112 `RESOURCE_REMOVAL`) — no crash, and every single
`RESOURCE_REMOVAL` magnitude logged fell correctly within `[0.2, 0.5]` (spot-checked the full run:
e.g. 0.211, 0.255, 0.339, 0.499 — no more 100%-of-pool clamping). One full PPO rollout + policy
update completed, `model_final.zip` saved. `RESOURCE_REMOVAL` is confirmed working correctly in
training, not just no-longer-crashing.

### Step 56 — `gym_env.py`: `phi(t)` redefined as a non-overlapping ("tumbling") batch mean, replacing the rolling window

**Why:** Walking the user through a pasted training-log episode surfaced a real design flaw:
`phi(t)` was a *rolling* mean over the last `ppi_window_n=20` completed traces (a `deque(maxlen=20)`).
Each new completion evicted only the single oldest trace from that window — so immediately after a
shock fires, the window is still ~80-95% pre-shock data, and stays that way until enough new
completions have rolled through to flush the old ones out one at a time. The user's diagnosis:
"having 16 old traces in the calculation does not make sense at all — of course we would not see
the effect of the shock," with the fix specified directly: compute the mean of the last 5 completed
traces, recomputed once per 5 new completions, with no two consecutive means sharing a trace.

**Done:** `ppi_window_n`'s semantics changed from "rolling window size" to "non-overlapping batch
size." New state: `self._phi_batch_buffer` (accumulates new completions not yet forming a full
batch), `self._current_phi` (the last fully-resolved batch's mean, held constant/piecewise-
step-function between resolutions), `self._phi_batches_completed` (count of batches resolved this
episode). The "Gather rewards" loop (where individual trace completions are processed one at a
time) appends each new cycle time to the buffer and, once it reaches `ppi_window_n`, computes the
mean, resets the buffer to empty, and increments the batch counter — no completion is ever counted
in two different `phi(t)` values. `_current_ppi()` now just returns `self._current_phi`.
`_warm_up()`'s stopping condition changed from "rolling window full" to "at least one batch has
resolved" (`self._phi_batches_completed < 1`). Added a new `ppi_window_n_override` constructor
parameter (mirroring the existing `p_star_override` pattern) so callers can use a smaller batch
(5) than whatever is on the artifact (every artifact currently says 20, inherited from the old
design) without editing the artifact file. Removed the now-dead `from collections import deque`
import.

Critically, at the warm-up→shock transition (where `total_reward`/`reward_count` were already
being zeroed so post-shock bookkeeping doesn't include warm-up), `self._phi_batch_buffer` is now
*also* cleared — any pre-shock completions not yet forming a full batch are discarded, so the
first post-shock `phi(t)` reading is built entirely from post-shock completions, never diluted by
leftover pre-shock data. `self._current_phi` itself is deliberately left untouched at this
boundary — the warm-up-established baseline is exactly what should be reported as the "before"
reference at shock onset.

### Step 57 — `gym_env.py`: recovery condition fixed to require a genuinely fresh post-shock reading

**Why:** Found while empirically testing Step 56, not before: with a smaller batch (5), it became
common for the shock's own duration to elapse *before* even one post-shock batch of 5 completions
resolved. The old recovery check (`shock ended AND phi(t) <= P*`) would then declare "recovered"
using whatever `phi(t)` last happened to read — which, in that situation, is still the stale
pre-shock baseline, not a genuine post-shock measurement. This would have quietly defeated the
entire point of Step 56: a "recovered" episode that never actually re-measured anything after the
shock.

**Done:** Added `self._phi_batches_completed_at_shock_onset`, snapshotted the moment the shock
fires. The recovery condition now additionally requires
`self._phi_batches_completed > self._phi_batches_completed_at_shock_onset` — i.e. at least one
new batch must have resolved since the shock fired — before `phi(t) <= P*` is even consulted.
Consistent with the project's existing "no backstop" philosophy (Step 41): if a fresh reading
takes a long time to materialise, the episode just keeps running until it does, same as it always
has for genuine recovery.

**Verified:** instrumented 4 seeded runs pre-fix (all showed `recovered=True` using a stale
pre-shock `phi(t)` value, 0 post-shock batches resolved) and the same 4 seeds post-fix (all now
require `batches_since_shock_onset=1` before recovering, with `phi(t)` values genuinely computed
from fresh post-shock data, e.g. 16,436 / 23,356 / 17,025 / 18,118 vs. `P*`=32,856).

### Step 58 — `calibrate_resilience_constants.py`, `compute_resilience_metrics.py`, `train_DRL.py`: `ppi_window_n` threaded through consistently

**Why:** `P*`/`cpl_base`/`phi_fix` are only meaningful relative to whatever `phi(t)` definition
computed them. Changing `phi(t)`'s batch size without also making every downstream script use the
*same* size would silently recreate an inconsistency identical in spirit to the ones already fixed
this project (mismatched pools, mismatched shock regimes, etc.).

**Done:** `calibrate_resilience_constants.py` gained `--ppi-window-n` (default `5`), threaded into
`build_env()`'s `gym_env` construction and recorded in `config_used.ppi_window_n`.
`compute_resilience_metrics.py` reads `ppi_window_n` back out of the loaded `--calibration` file
(same pattern already used for `P_star`) rather than defaulting independently — falls back to 20
with a loud warning if an older calibration file (pre-dating this redesign) doesn't have the field
recorded. `train_DRL.py` does the same: reads `ppi_window_n` from the calibration file and passes
it to both `env_simulator` and `eval_env` construction via `ppi_window_n_override`, with the same
loud-warning fallback for old files.

**Verified:** Consulta and BPI2017 both recalibrated (Step 59) and both new calibration files show
`"ppi_window_n": 5` recorded in `config_used`, confirming the value round-trips correctly from the
calibration script through to the file used by both `compute_resilience_metrics.py` and (once
training resumes) `train_DRL.py`.

### Step 59 — Consulta and BPI2017 recalibrated and re-validated under the new `phi(t)` definition

**Why:** Steps 56-58 changed what `phi(t)` means; the existing `consulta_calibration.json` and
`bpi2017_calibration.json` (and the resilience-metrics results computed from them) were stale,
calibrated against the old rolling-window metric.

**Done:** Re-ran `calibrate_resilience_constants.py` for both, using the exact same validated
shock parameters as before (unchanged — magnitude=6, Consulta duration=72,000s/20h, BPI2017
duration=21,600s/6h; only `--ppi-window-n 5`, new since Step 58's default, was added) and
`compute_resilience_metrics.py` for both (same magnitude grid `5,6,7,8`, seeds `0,1`, same
durations as calibration).

**Results — both `cpl_base` values remain large and non-zero, confirming the shock still has a
real effect under the new metric:**

| | Consulta (old → new) | BPI2017 (old → new) |
|---|---|---|
| `P_star` | 32,856.0 → 32,856.0 (unchanged) | 3,816.51 → 3,816.51 (unchanged) |
| `cpl_base` | 33,710,522,992.9 → **53,218,071,893.7** (+58%) | 167,982,158.7 → **131,704,591.7** (−22%) |
| `phi_fix` | 236,243.6 → 242,446.96 (~unchanged) | 8,092.11 → **11,538.07** (+43%) |

Resilience-metrics runs (magnitude 5-8 × 2 seeds each) show a dramatic improvement over the old
metric's results in both datasets: seed-to-seed spread at the same magnitude drops from the old
~65x (BPI2017, magnitude=6, old metric) to **1.4-1.9x** across most of the grid for both datasets
— the old spread was metric-noise, not genuine variance. Consulta shows a clean, monotonically
increasing CPL/RT with magnitude in both seeds independently. BPI2017 shows a real, reproducible
threshold effect between magnitude 6 and 7 (RT roughly triples in *both* seeds independently:
2.07h→7.76h and 2.34h→7.58h), with wider seed divergence specifically at the highest tested
magnitude (8) — plausibly a genuine effect of which resources get hit at the most extreme shock
tested, not pure noise, given how tight the spread is at every other magnitude. Full results
written to `Module 1/artifacts/consulta_resilience_metrics.json` and
`Module 1/artifacts/bpi2017_resilience_metrics.json`. See
`Training_Readiness_Implementation_Summary.md` §50-52 for the full narrative and tables.

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
| 20 | `Module 1/empirical_preprocessing.py` | Added wide-format true-PT detection (`start_timestamp` column, `_compute_true_pt_wide()`) |
| 21 | `Module 1/empirical_preprocessing.py` | Strip activity-name whitespace (fixes BPI2017 `'W_Shortened completion '` vs. Petri net's `'W_Shortened completion'` mismatch) |
| 22 | `Module 1/empirical_preprocessing.py` | Added `activity_merge_map`/`exclude_activities` params (Consulta fake-resource fix); synthetic `"__NO_RESOURCE__"` placeholder injected into `resource_processing_times` for every `activities_at_risk` entry, excluded from `resource_tier_map` |
| 23 | `Module 1/empirical_preprocessing.py` | Added `P_star`/`normalization_cycle_time`/`ppi_window_n` artifact fields; regenerated all 4 artifacts (`bpi2017.json`, `bpi12w.json`, `helpdesk.json`, new `consulta.json`) |
| 24 | `gym_env.py` | Added `self.ppi_history` rolling-PPI tracking (reset per episode) + `_current_ppi()` helper |
| 25 | `gym_env.py` | Load `P_star`/`ppi_window_n`/`normalization_cycle_time` from artifact, overriding the hardcoded constant |
| 26 | `train_DRL.py` | Hardcoded `postpone=False` at every `gym_env(...)` call site (CLI arg still parsed, now ignored) |
| 27 | `gym_env.py` / `process.py` | Synthetic placeholder resource wired into `self.resources` (once, at construction) and `SimulationProcess._resources` (every `reset()`, as a `RoleSimulator(capacity=math.inf)`) |
| 28 | `train_DRL.py` | Fixed FIFO `cal_env` bug: `reward_function='RF2'` (was `'inverse_CT'`, which silently kept `cpl_accumulated` at `0.0`) |
| 29 | `train_DRL.py` | Added `p_cpl_fraction = 0.5`; `cpl_budget = p * cal_env.cpl_accumulated` |
| 30 | `train_DRL.py` | Added random-policy `phi_fix` calibration loop (`K=5` episodes, worst PPI observed) |
| 31 | `gym_env.py` | Added `phi_fix`/`beta` params; implemented the `'RF1'` reward branch (r1 velocity + r2 deviation, beta-combined) |
| 32 | `gym_env.py` | Rewrote RF2: `d(t)`-based `B_k`/`u_k`/`v_k`, dropped shock-flag gate, Lagrange update via `v_k`, fixed mislabeled `u_k` observation feature, fixed missing `RF1` branch in the episode-end `total_reward_end` block |
| 33 | `inter_trigger_timer.py` | Pandas/prophet compat shim: strip trailing `Z` from serialized-model timestamps before `model_from_json` (unblocked all Prophet-based arrival generation under current library versions) |
| 34 | Validation (no file changes) | Phase 14 manual-stepping pilot on BPI2017 — all checks pass; found and drove Steps 21/32's fixes; surfaced (did not fix) a `next_decision_moment()` livelock risk under shocks + a finite resource pool |
| 35 | `shock.py`, `gym_env.py`, `train_DRL.py` | Single-shock-per-episode redesign: `fire_single_shock()` replaces the recurring scheduler; hidden FIFO warm-up in `reset()` establishes `phi(t)` baseline before firing; recovery-based `terminated` + real `truncated` backstop in `step()`; calibration loops exit on `terminated or truncated` — resolves Step 34's livelock risk at its root |
| 36 | `gym_env.py` | Consulta activity-merge-map wiring: `activity_merge_map`/`_canonical_activity()` translate native Petri-net labels to Module 1's merged label at every token-vs-action-space comparison point; `_resource_events` rebuilt post-construction in `reset()` to match — fixes a pre-existing bug that made Consulta unsimulatable in any configuration |
| 37 | `gym_env.py`, `train_DRL.py` | Reproducibility: `reset(seed=...)` now actually seeds `random`/`np.random` (was accepted but ignored); `cal_env`/`rand_env` calibration loops pass explicit seeds |
| 38 | `shock.py` | Activity-scoped `RESOURCE_REMOVAL` (new default): draws from one activity's own eligible-resource pool instead of the whole system pool, so the same magnitude is a much bigger fractional hit; `'global'` (old behaviour) kept for comparison |
| 39 | `process.py`, `event_trace.py`, `shock.py`, `inter_trigger_timer.py` | Fixed `DEMAND_SPIKE`, which had never worked on any dataset: `Token.inter_trigger_time()` is now interruptible and recomputes its remaining wait from the live `arrival_multiplier`; `shock.py` interrupts all pending arrivals at onset and offset; fixed a related `_start_trace` cycle-time-accounting bug this surfaced; removed the old always-no-op multiplier division in `inter_trigger_timer.py` |
| 40 | `Module 1/empirical_preprocessing.py` | Removed `aggregate()`'s fallback re-admission of low-support candidates for activities with zero eligible resources after the `n_min` filter — was letting a 2-observation outlier (Consulta's `'1343'` on `"Transferir Creditos"`, 16.6-day median) become the sole always-selected resource for an activity, dominating simulated cycle times; such activities now correctly fall through to the existing placeholder-resource path instead; regenerated `consulta.json` |
| 41 | `gym_env.py` | Removed the recovery backstop (`max_recovery_wait_s`, `5×duration`) entirely — episodes now terminate only via genuine recovery or trace-population exhaustion, never a time-based cutoff |
| 42 | `gym_env.py` | Added `pool_override_path`/`p_star_override` constructor params — loads a curated resource pool and/or a fixed `P*` in-process, folding this session's scratchpad monkey-patch pattern into real code |
| 43 | new `calibrate_resilience_constants.py` | Checked-in script computing and persisting `P_star` (2×pooled median), `cpl_base` (mean `cpl_accumulated` pooled across seeds, not a single episode), and `phi_fix` for a given (dataset, pool) pair |
| 44 | `train_DRL.py` | Removed inline FIFO/random-policy calibration; loads the persisted calibration file instead (new CLI args: `pool_override_path`, `calibration_path`), failing loudly on a missing or mismatched file |
| 47 | new `build_helpdesk_engine_bundle.py`, `verify_helpdesk_bundle.py` | Built Helpdesk's RIMS_DRL engine bundle from scratch (`.pnml` via Inductive Miner, `input_Helpdesk.json`, minimal `_dispr_meta.json`) — no pre-existing bundle shipped for this dataset |
| 48 | `inter_trigger_timer.py` | Fixed the `'distribution'` arrival mode: `self._previous += self._interval` was commented out, so every arrival was silently scheduled at `t=0` — found via Helpdesk bundle verification |
| 49 | new `concurrency_oracle_estimator.py`, `validate_concurrency_oracle.py` | Directly-Follows concurrency-oracle start-time estimator + BPI2017/Consulta ground-truth validation (real improvement over naive, but unreliable on human-scale activities for both) |
| 50 | new `build_helpdesk_co_pool.py`, `calibrate_helpdesk_duration_scale.py` | Helpdesk duration calibration: single global scale factor `k=0.36478` found via bisection search against the real log's median cycle time (citation-backed methodology, see `Duration_Calibration_Methodology.md`) |
| 51 | `calibrate_resilience_constants.py` | Added `--p-star-formula {median,mean}` — dataset-scoped override for heavily bimodal cycle-time distributions (PurchasingExample) |
| 52 | `calibrate_resilience_constants.py` | `--pool` made optional (was `required=True`) — some datasets' resource distributions don't need a curated pool |
| 53 | `train_DRL.py` | Removed dead `from gymnasium.wrappers import normalize` import — incompatible with installed gymnasium 1.2.3, crashed training on startup |
| 54 | `train_DRL.py` | Train-time `shock_types` restricted to `['DEMAND_SPIKE']` only (interim) — `RESOURCE_REMOVAL` shares a magnitude range meant for `DEMAND_SPIKE`'s multiplier semantics, causing 100%-of-pool removal and a stranded-token crash |
| 55 | `shock.py`, `train_DRL.py` | Real fix superseding Step 54: new `resource_removal_magnitude_range` field on `ShockConfig` (default `(0.2, 0.5)`), used only for `RESOURCE_REMOVAL` draws; `RESOURCE_REMOVAL` re-enabled in `train_DRL.py` |
| 56 | `gym_env.py` | `phi(t)` redefined: non-overlapping tumbling-batch mean (default batch size 5) replacing the old rolling-window-of-20 mean; buffer cleared at the warm-up→shock boundary so the first post-shock reading is never diluted by pre-shock data |
| 57 | `gym_env.py` | Recovery condition fixed: now requires a genuinely new post-shock `phi(t)` batch to have resolved, not just "shock's over and the last-known (possibly stale) reading is <= P\*" |
| 58 | `calibrate_resilience_constants.py`, `compute_resilience_metrics.py`, `train_DRL.py` | `ppi_window_n` (tumbling-batch size) threaded consistently through calibration, resilience-metrics, and training — read back from the calibration file everywhere, never re-defaulted independently |
| 59 | (no code change — recalibration run) | Consulta and BPI2017 recalibrated and re-validated (resilience-metrics grid) against the new `phi(t)` definition |

**Milestones:**
- Steps 1–4: code runs without TensorFlow/Prophet
- Steps 5–8: DRL training loop works with empirical durations
- Steps 9–13: shock injection active
- Steps 14–16: extended observation and RF2 reward complete
- Step 17: full training harness wired
- Steps 18–19: Module 4 feed complete (post-training only)
- Steps 20–23: Module 1 artifact ready for RF1/RF2 (P*, C, PPI window, placeholder resource, wide-format PT, Consulta fix) across all 4 target datasets
- Steps 24–27: Modules 2-3 shared plumbing for RF1/RF2 (rolling PPI, artifact fields, postpone removed, placeholder wiring)
- Steps 28–30: calibration fixed/added (`CPL_base`, `phi_fix`)
- Step 31: RF1 implemented
- Step 32: RF2 rewritten to spec
- Step 33: unrelated environment compat fix (Prophet arrival generation)
- Step 34: BPI2017 validation pilot passes; `next_decision_moment()` livelock risk flagged as open (not fixed)
- Step 35: livelock risk resolved via single-shock-per-episode redesign; shock-enabled validation passes on all tested scales
- Step 36: Consulta unblocked (activity-merge-map wiring) — all 4 target datasets now simulatable, at least on the full historical resource pool
- Step 37: reset(seed=...) reproducibility fixed; calibration loops (`CPL_base`, `phi_fix`) now reproducible
- Phase 17 (not numbered steps): resource-pool restriction and simulator-native `P*` — validated methodology with concrete candidate numbers, not yet written into production code or artifact files; see `Training_Readiness_Implementation_Summary.md` for the full narrative and current numbers
- Step 38: `RESOURCE_REMOVAL` now scoped to one activity's own pool by default, so a given magnitude is a meaningfully bigger capacity hit
- Step 39: `DEMAND_SPIKE` fixed — confirmed non-functional on every dataset prior to this step, now verified (direct arrival-timestamp instrumentation) to genuinely compress arrivals, and produces real, measurable `n_pending` backlog and non-trivial recovery times end-to-end on all 3 simulatable datasets — the shock layer is, for the first time in this project, doing something visibly real
- Step 40: real Consulta data-quality bug fixed (`aggregate()` fallback re-admission removed) — resolves a multi-million-second `phi(t)` blowup present in every prior Consulta result, shocked or not; `consulta.json` regenerated
- Phase 20 (not numbered steps): resilience-metric methodology matured substantially — least-used baseline policy (real fix), `max_recovery_wait_s`/`N_TRACES` findings (both worked around in test scripts, not fixed in `gym_env.py` itself), a working zero-specialist pool-construction algorithm, `P*` locked to a fixed `2×median`, and a checked-against-reality shock-magnitude ceiling — culminating in a clean, controlled three-dataset resilience comparison (Consulta and BPI2017 both clearly shock-fragile; BPI12W genuinely more robust even under matched lean staffing). See `Training_Readiness_Implementation_Summary.md` §27-35 for the full narrative and numbers.
- Steps 41-44: Phase 20's methodology wired into production for real — backstop removed from `gym_env.py`, pool/`P*` overrides added, a real calibration script created, `train_DRL.py` now loads a persisted calibration file instead of recomputing inline. Consulta's calibration file is complete and correct (including a fix to `cpl_base`, which a single-seed design had been silently zeroing out).
- Step 45: `--n-traces-baseline` added to `calibrate_resilience_constants.py` — isolates the one genuinely slow stage (the no-shock `P*` baseline, which can never recovery-terminate early) from the shocked stages (`cpl_base`/`phi_fix`, which already terminate on recovery in ~90-130 traces). BPI2017's calibration resumed and completed: `P_star=3816.5s`, `cpl_base=167,982,158.72`, `phi_fix=8092.11s`. Both Consulta and BPI2017 are now calibration-complete and training-ready — see `Training_Readiness_Implementation_Summary.md` §37-40.
- Step 46: `train_DRL.py`'s train-time `shock_config` aligned to the calibrated/eval shock — `magnitude_range` `(0.2, 0.5)` → `(5.0, 8.0)`, `duration_range` `(1800.0, 7200.0)` → fixed `(21600.0, 21600.0)`. Closes the last flagged gap: training and evaluation now see the same disruption regime — see `Training_Readiness_Implementation_Summary.md` §41.
- Steps 47-48: Helpdesk's RIMS_DRL engine bundle built from scratch (no pre-existing bundle shipped for this dataset, unlike BPI2017/Consulta/PurchasingExample) — surfaced and fixed a real, previously-unexercised engine bug in the `'distribution'` arrival mode along the way.
- Step 49: concurrency-oracle-based start-time estimation implemented and validated against BPI2017/Consulta ground truth before ever being trusted on Helpdesk — a real improvement over the naive sojourn approximation, but not reliable enough on its own for human-scale activities.
- Step 50: Helpdesk's duration-estimation problem resolved via a citation-backed single global scale-factor calibration (median fit +14.8%, mean held-out +8.2%, p95 held-out +36.8% — documented limitation) rather than continuing to chase a better per-event heuristic.
- Steps 51-52: `calibrate_resilience_constants.py` generalized (`--p-star-formula`, optional `--pool`) to support PurchasingExample's different resource/distribution characteristics without touching any already-validated dataset's behaviour.
- Steps 53-55: first end-to-end training smoke test run (Consulta) — found and fixed a dead-import crash (Step 53) and a `RESOURCE_REMOVAL` magnitude-range bug, first via an interim `DEMAND_SPIKE`-only restriction (Step 54) then a real fix giving `RESOURCE_REMOVAL` its own correctly-scaled range (Step 55). Both shock types now verified working together in training (213 shocks fired cleanly, all `RESOURCE_REMOVAL` magnitudes correctly bounded). See `Training_Readiness_Implementation_Summary.md` §48-49 for the full narrative, and §45/§47 for Helpdesk's and PurchasingExample's current (not-yet-training-ready) status.
- Steps 56-57: `phi(t)` redefined from a rolling window to a non-overlapping tumbling-batch mean (default size 5), after the user identified that the old design left a shock's aftermath diluted by up to 95% pre-shock data for a long stretch after onset. Testing this surfaced a second, related bug — recovery could be declared using a stale pre-shock reading if the shock ended before a single post-shock batch resolved — fixed in the same pass by requiring a genuinely fresh post-shock batch before recovery can trigger.
- Step 58: the new `ppi_window_n` threaded consistently through calibration, resilience-metrics, and training scripts — same discipline already applied to `P*`/`cpl_base`/`phi_fix`/pool consistency.
- Step 59: Consulta and BPI2017 recalibrated and re-validated against the new `phi(t)`. Both `cpl_base` values remain large and non-zero (Consulta +58%, BPI2017 -22% vs. the old metric). Resilience-metrics results improved dramatically: seed-to-seed spread dropped from ~65x (old metric, BPI2017) to 1.4-1.9x across most of the magnitude grid for both datasets. Consulta shows a clean monotonic dose-response with magnitude; BPI2017 shows a real, reproducible threshold effect between magnitude 6 and 7 (RT roughly triples in both seeds independently). See `Training_Readiness_Implementation_Summary.md` §50-52 for the full narrative and tables.
