# RIMS_DRL Modification Record

This document records every change made to the `RIMS_DRL-main` codebase, in execution order. All 34 steps are complete. Each entry names the file, what was changed, and why.

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
