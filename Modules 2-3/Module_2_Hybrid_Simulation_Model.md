# Module 2 – The Hybrid Simulation Model: Implementation Specification

## Purpose

Build the business process simulator that the DRL agent (Module 3) interacts with at training
and evaluation time. This module extends **RIMS (Runtime Integration of Machine Learning and
Simulation)**, the simulation framework from Meneghello et al. (2023, 2024), in two specific
ways:

1. Replace RIMS's ML-based duration model with Module 1's **empirical per-resource
   processing times**, so the simulator times each activity from the historical performance of
   the specific resource assigned to it.
2. Add a **shock injection mechanism** that perturbs the simulated process at runtime.

The agent operates at the **individual-resource level** (per the updated TDR): each action is
the assignment of one specific resource to the pending activity. Performance tiers
(`resource_tier_map`) are **not** used by this module for allocation — they are consumed
silently by Module 3 (cluster-layer observation) and Module 4 (distillation). Module 2 only
needs the individual-resource pool and the empirical processing times.

This module does **not** implement RIMS from scratch — it wraps/extends the existing RIMS
codebase/methodology. RIMS's discrete-event mechanics (queueing, resource busy/free tracking,
case arrivals, and control-flow routing) are **inherited unchanged**; only the *source of
activity durations* is replaced, and the shock mechanism is added on top. Where RIMS internals
are not literally specified in the source proposal, this spec marks them as "inherit from RIMS"
rather than inventing new behavior.

---

## 1. Dependencies / Starting Point

- **Base simulation engine:** RIMS, per Meneghello et al. (2023) — a hybrid of Data-Driven
  Process Simulation (DDPS) and Discrete Event Simulation (DES). In stock RIMS, the *time
  perspective* (activity durations) comes from an ML predictor. **This module replaces that
  duration source with an empirical lookup (Section 2).** RIMS's DES backbone — queues, resource
  busy/free state, arrival generation, and routing — is kept as-is.
- **Module 1 artifacts consumed:**
  - `resource_processing_times`: for each `(resource, activity)` pair, the capped median
    historical processing time (Module 1, Stage 1). This is the duration source.
  - `resource_tier_map`: per-resource tier label per activity (Module 1, Stage 2). Used here
    **only** to (a) determine eligibility — a resource with no entry for an activity has never
    performed it and is ineligible — and (b) optionally scope resource-removal shocks
    (Section 3). It is **not** used for timing or for the agent's action space.
- **Implementation requirement before writing extension code:** locate/clone the RIMS reference
  implementation (or reimplement its public interface if no usable codebase exists) and identify
  its extension points for: (a) resource pool definition, (b) the per-decision-epoch simulation
  loop, (c) the duration/time-sampling hook, and (d) the arrival-process configuration. These
  are the points this module modifies.

---

## 2. Extension 1 — Individual-Resource Pool with Empirical Durations

**Goal:** the simulator's resource pool is the set of individual resources from the log, each
timed by its own historical processing times, so that reassigning work to a different resource
changes the duration in exactly the way the historical data supports.

### 2.1 Resource pool construction
- Build one pool entry per individual resource observed in the log (after Module 1 exclusions):
  ```
  ResourcePool[resource_id] = {
      "capacity": 1,                              # one worker, one task at a time
      "eligible_activities": <activities this resource has an entry for
                              in resource_processing_times>,
      "available": True                           # runtime busy/free flag
  }
  ```
- `capacity = 1` per individual resource is the natural analogue of RIMS's per-worker busy/free
  tracking. A resource is decremented to busy on assignment and restored on completion.
- Total pool size must equal the number of distinct resources in the log, minus those dropped by
  Module 1 (zero/negative durations). No resource is silently lost; every exclusion is logged.

### 2.2 Processing-time source (replaces RIMS's ML duration model)
- When the simulator needs the duration of `activity` performed by the assigned `resource_id`,
  it looks up:
  ```
  duration = resource_processing_times[resource_id][activity]
  ```
  i.e. the capped empirical median computed in Module 1. There is **no** parametric distribution
  fit and **no** ML regressor for durations — this is a deliberate simplification relative to
  stock RIMS.
- **Load-independence is intentional and must be stated as a modelling assumption.** Because the
  duration depends only on `(resource, activity)` and not on runtime state (queue length, whether
  a shock is active, time of day), a resource always takes its historical median time regardless
  of pressure. This is what makes the simulator robust in the shock regime: congestion is *not*
  predicted by any model that could extrapolate badly — queues and waiting times **emerge** from
  the DES mechanics (arrivals + these fixed service times + finite capacity). The trade-off is
  that any real-world tendency for service times to change under load (rushing, fatigue, rework)
  is not represented. Implication for evaluation: since the assumption applies identically to the
  agent and all baselines, the **paired ranking** of decision-makers is unaffected; only the
  **absolute** resilience figures may be optimistic. This belongs in the thesis's
  threats-to-validity discussion.
- **Optional sampling variability (open decision).** As defined, `resource_processing_times`
  stores a single median per pair, so the duration is deterministic per `(resource, activity)`.
  If stochastic service times are desired, Module 1 must additionally retain the per-pair array
  of (capped) historical durations, and this module would sample from that empirical array
  instead of using the point median. Either choice remains load-independent. Default:
  deterministic median unless the sampled variant is explicitly enabled.

### 2.3 Assignment at simulation time
- Whenever stock RIMS would assign an individual worker to a task, this module instead:
  1. Reads the action handed down by Module 3 (the chosen `resource_id` for the pending activity).
  2. Verifies eligibility and availability (this is guaranteed upstream by action masking, but
     re-checked defensively).
  3. Marks `resource_id` busy for the looked-up `duration` (2.2), then restores it on completion.
- **Action-masking dependency:** the simulator must expose, at each decision point, the set of
  resources that are eligible for the pending activity (have a surviving
  `resource_processing_times` entry) **and** currently available.
  This feasibility signal is required by Module 3's action masking and must be a queryable
  method — see Section 4 (`get_feasible_resources`).

---

## 3. Extension 2 — Shock Injection Mechanism

**Goal:** at simulation runtime, probabilistically trigger one of two shock types, sustain it for
a defined duration, and expose its state to the agent's observation (consumed in Module 3, but
the shock *state itself* lives here).

### 3.1 Shock types (exactly two, per the proposal)
```
ShockType.RESOURCE_REMOVAL:
    params: { target: <selector>, fraction_m: float in (0, 1] }
    effect: while active, a fraction_m of the *currently-available* resources matching
            `target` are marked unavailable for new assignments. Resources already busy are
            allowed to finish their current task (no pre-emption) unless a different policy is
            explicitly chosen and documented. On shock end, removed resources become available
            again.

            `target` selects which resources the removal applies to. Because the design no
            longer has global role groups, the selector is one of:
              - the whole active pool (default), or
              - a named subset of resource IDs (e.g. a department in the Sepsis log), or
              - a tier-scoped subset (e.g. resources whose resource_tier_map[·][activity] = fast),
                to simulate "losing the best people" for an activity.
            The proposal's "30% reduction in available Generalists" maps onto a named/tier-scoped
            subset with fraction_m = 0.3.

ShockType.DEMAND_SPIKE:
    params: { fraction_m: float > 0 }
    effect: while active, the case arrival rate (however RIMS natively samples new-case arrivals)
            is multiplied by (1 + fraction_m).
```

### 3.2 Shock triggering — Poisson process
*(The Module 2 text in the source proposal says the triggering distribution "is yet to be
decided." The worked-out model appears later under "The Shock and Its Distribution" and is the
correct mechanism to implement — restated here since Module 2's own text underspecifies it.)*

- Shock **onsets** are a homogeneous Poisson process with rate `λ`: inter-shock intervals are
  i.i.d. exponential with rate `λ`, onsets independent.
- Implementation:
  ```python
  next_shock_delay = numpy.random.exponential(scale=1 / lambda_rate)
  ```
  scheduled relative to current simulation time, in the simulator's native time unit.
- `λ` is a configurable hyperparameter (`shock_rate_lambda`), tuned per event log; no numeric
  value is given in the proposal.
- Shock **type**, **size** (`fraction_m`), and **target** (for resource removal) are each sampled
  from a configurable range at each onset — see 3.3.
- Shock **duration**: each shock persists for a configurable number of timesteps
  (`shock_duration_min`, `shock_duration_max`, sampled uniformly, or a fixed constant). Not pinned
  down numerically in the proposal — an open tuning decision.

### 3.3 Training-time shock variety (required, not optional)
If the agent only ever sees one fixed shock configuration, it overfits to that scenario.
Therefore, **during training rollouts**, shock generation must vary across:
- Both shock types (resource removal and demand spike) — sampled, not fixed per run.
- A range of magnitudes (`fraction_m` from a configured range, e.g. uniform on `[m_min, m_max]`).
- Random onset times (the Poisson process provides this).
- Occasionally **overlapping/compound shocks** — a new shock may be triggered while a previous
  one is still active. Overlap is permitted **only** in training mode.

### 3.4 Evaluation-time shock control (required, distinct from training mode)
At **evaluation/test time**, shocks must NOT overlap, so that a clean before/during/after
recovery curve is well-defined. A `mode: "train" | "eval"` flag governs this:
- `mode="train"`: full random variety as in 3.3, overlaps allowed.
- `mode="eval"`: **exactly one** shock per run, at a known, configured time, with a known,
  configured type, magnitude, and target — fully deterministic, not sampled from the Poisson
  process. This produces the baseline → hit → recovery curve used for the Kraus et al.
  (2024)-style resilience graph (Resilience Metrics / Module 4).

### 3.5 Shock state exposed to the environment
At every timestep, the shock mechanism exposes (for Module 3's observation space):
- `shock_active: bool` — 1 while any shock is in effect, 0 otherwise.
- The affected resources and magnitude are reflected implicitly through the actual
  availability/arrival-rate changes applied in 3.1–3.2, not necessarily as separate observation
  features unless Module 3 specifies otherwise.

---

## 4. Required Public Interface (what the Module 3 / Gymnasium wrapper calls)

```python
class ShockInjectedRIMSSimulator:
    def reset(self, mode: Literal["train", "eval"], eval_shock_config: dict | None = None):
        """Re-initialize simulation state, the individual-resource pool (from Module 1's
        resource_processing_times), and the shock generator in
        the given mode."""

    def step(self, resource_assignment: dict[str, str]):
        """Advance the simulation by one decision epoch given the chosen resource_id for the
        pending activity. Returns the updated simulation state."""

    def get_feasible_resources(self, activity: str) -> list[str]:
        """Return resource_ids that are (a) eligible for `activity` — i.e. have a
        resource_processing_times entry — AND (b) currently
        available. Used by Module 3 action masking."""

    def get_shock_state(self) -> dict:
        """Return at minimum {'shock_active': bool}."""

    def get_ppi_snapshot(self) -> dict:
        """Return the current value(s) of the Process Performance Indicator(s) being tracked
        (e.g. running lead time / cycle time), for resilience-metric computation downstream."""
```

Note: Module 2 exposes only individual resources. The translation of a chosen resource into its
performance tier (for the cluster-layer behavioral log) is done by Module 3 using
`resource_tier_map`; it is not Module 2's responsibility.

---

## 5. Validation / Sanity Checks

- **Pool completeness:** the individual-resource pool count equals the number of distinct
  resources in the log, minus Module 1 exclusions (zero/negative durations). Every dropped
  resource/pair is logged.
- **Eligibility integrity:** `get_feasible_resources` never returns a resource that is busy or
  that lacks an entry for the activity.
- **Eval-mode control:** under `mode="eval"`, exactly one shock fires per run and no overlap
  occurs across repeated `reset()` calls with different seeds.
- **Train-mode variety:** under `mode="train"`, over many episodes, observed shock-type
  frequencies, magnitude distribution, and overlap rate roughly match the configured sampling
  ranges (statistical smoke test, not strict equality).

---

## 6. Open Implementation Decisions (flag before coding)

- Numeric Poisson shock rate `λ` — not given; set/tune per event log.
- Shock duration — fixed vs. sampled range; not numerically specified.
- Range of `fraction_m` magnitudes for training — not numerically specified.
- Deterministic median vs. sampling from the empirical duration array (2.2) — depends on whether
  Module 1 retains per-pair samples; default deterministic.
- Resource-removal targeting scope (whole pool vs. named subset vs. tier-scoped) and its default.
- Pre-emption policy for resource removal — the "do not pre-empt in-progress tasks, block only new
  assignments" default is the more realistic choice but is not explicitly stated in the proposal;
  confirm.
- Whether RIMS's native arrival generator can be cleanly rate-scaled for `DEMAND_SPIKE`; inspect
  the chosen RIMS reference implementation before finalizing 3.1.
