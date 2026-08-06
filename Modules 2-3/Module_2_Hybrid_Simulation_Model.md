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

### 3.2 Shock triggering — single shock per episode (supersedes the original Poisson-process design)
*(The Module 2 text in the source proposal originally left the triggering distribution "yet to be
decided," and an earlier version of this spec adopted a recurring homogeneous Poisson process —
restated below for what it was, since the actual implementation has since diverged from it.)*

**As implemented:** each episode fires **exactly one shock**, right after a hidden warm-up phase
(the simulator runs undisturbed for a short initial period first, via a least-used/FIFO-style
policy internal to `reset()`, so the episode doesn't start mid-void). This replaced the original
recurring Poisson-process scheduler after the latter was found to cause a livelock: a new shock
could fire while a previous one's recovery was still in progress, and the environment's
decision-epoch loop (`next_decision_moment()`) could stall indefinitely under compounding
disruption. See `RIMS_DRL_Modification_Plan.md` Step 35 for the incident and fix.

- Shock **type**, **magnitude**, and **duration** are still sampled per episode from configured
  ranges (train mode) or fully fixed (eval mode) — see 3.3/3.4. Only the *count* and *scheduling
  mechanism* changed: one shock, deterministically placed right after warm-up, not a recurring
  Poisson arrival process through the whole episode.
- `λ` (`shock_rate_lambda` / `lambda_rate`) is **no longer read** by this flow — it only mattered
  for the old recurring scheduler's inter-shock interval. Left as a vestigial config field in case
  a recurring scheduler is reintroduced elsewhere; not part of the single-shock path.
- Shock **duration**: a configurable range (train mode) or a fixed value (eval mode, and — as of
  this project's own training-time alignment fix — also fixed in train mode, matched to the
  calibrated/eval duration; see Module 3 §4). Concretely, `21,600s` (6h) for BPI2017 and Consulta's
  calibration/eval runs.

### 3.3 Training-time shock variety (still required, mechanism changed)
If the agent only ever sees one fixed shock configuration, it overfits to that scenario.
Therefore, **during training rollouts**, the single shock fired per episode still varies across:
- Both shock types (resource removal and demand spike) — sampled per episode, not fixed.
- A range of magnitudes (`fraction_m`/`magnitude` from a configured range). As implemented and
  validated: `[5, 8]` for `DEMAND_SPIKE` (a 6x-9x arrival-rate multiplier — `1 + magnitude`),
  established as the realistic ceiling by checking against real event-log arrival volatility
  (`Training_Readiness_Implementation_Summary.md` §32); magnitude 20-100 was tested early on and
  found to have no connection to plausible reality.
- **Not** overlapping/compound shocks — with exactly one shock per episode, by construction there
  is nothing for a second shock to overlap with. The original spec's "overlapping shocks permitted
  only in training mode" no longer applies; there is no scenario, train or eval, where two shocks
  are simultaneously active.

### 3.4 Evaluation-time shock control (still distinct from training mode, narrower gap now)
At **evaluation/test time**, exactly one shock fires, at a known, fully fixed type/magnitude/
duration/timing — unchanged from the original intent. The gap between train and eval mode is
narrower than originally specified, now that both are single-shock: train mode samples
type/magnitude per episode from a configured range (duration currently fixed, matched to eval —
see 3.2); eval mode fixes all of type/magnitude/duration/onset-timing to one exact configuration.
This produces the baseline → hit → recovery curve used for the Kraus et al. (2024)-style
resilience graph (Resilience Metrics / Module 4) and for the TTI/MPD/RT/CPL metrics computed by
`compute_resilience_metrics.py`.

**Episode termination — no time-based cap.** Neither train nor eval episodes are cut off by a
wall clock or a "give up waiting for recovery" backstop (an earlier `max_recovery_wait_s`
safeguard was found to truncate episodes before genuine recovery — e.g. one episode reported as
"never recovers," then confirmed to recover at 7.3 days once the cap was removed — and was deleted
entirely). An episode ends only when `φ(t)` genuinely drops back to `<= P*` after a shock, or the
trace population is exhausted, however long that takes.

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

- ~~Numeric Poisson shock rate `λ`~~ — **moot.** The recurring Poisson scheduler was replaced by
  single-shock-per-episode (3.2); `λ` is no longer read by this flow.
- ~~Shock duration~~ — **resolved.** Fixed at `21,600s` (6h) for the validated BPI2017/Consulta
  calibration and resilience-metric runs (dataset-specific; Consulta's own calibration shock used
  `72,000s`/20h — see `RIMS_DRL_Modification_Plan.md` Phase 21).
- ~~Range of `fraction_m` magnitudes for training~~ — **resolved.** `DEMAND_SPIKE`: `[5, 8]`
  (validated realistic ceiling, checked against real event-log arrival volatility —
  `Training_Readiness_Implementation_Summary.md` §32). `RESOURCE_REMOVAL`'s standing under the
  same fully-corrected methodology is not yet re-established (§39 of that document) — treat its
  magnitude range as still open.
- Deterministic median vs. sampling from the empirical duration array (2.2) — **still the
  original default** (deterministic median); not revisited this project.
- ~~Resource-removal targeting scope~~ — **resolved.** Activity-scoped by default (a given
  magnitude now removes a fraction of one activity's own eligible pool, not the whole resource
  pool), so a fixed magnitude is a meaningfully bigger capacity hit — see Step 38.
- Pre-emption policy for resource removal — **unchanged from the original default** (no
  pre-emption of in-progress tasks); not revisited this project.
- ~~Whether RIMS's native arrival generator can be cleanly rate-scaled for `DEMAND_SPIKE`~~ —
  **resolved: yes**, via interrupt-based arrival rescaling, confirmed by direct arrival-timestamp
  instrumentation (Step 39). This shock type was found to be silently non-functional before that
  fix — every result predating it should be treated as untrustworthy for `DEMAND_SPIKE`
  specifically.
