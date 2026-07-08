# Module 3 – DRL Environment and Agent: Implementation Specification

## Purpose

Wrap Module 2's simulator in a standard Gymnasium interface and train a PPO agent to learn
resilient resource allocation policies under exogenous shocks. The agent operates at the
**individual resource level** — it assigns specific resources to specific pending activities.
It never sees or reasons about performance tiers.

In parallel, the Gymnasium environment maintains a **cluster-layer observation** at every
decision step using Module 1's `resource_tier_map`. This cluster-layer stream is stored
silently alongside the agent's decisions and forms the behavioral log that Module 4 consumes
for policy distillation. The agent is entirely unaware of this parallel layer.

---

## 1. Connection to Module 1

Module 1 produces two artifacts that Module 3 loads at environment initialization:

**`resource_processing_times`** — passed directly to Module 2. When the agent assigns
resource `r` to activity `a`, Module 2 samples the duration from `r`'s processing time for
`a`. Resources with no entry for a given activity have never performed it and are ineligible
for assignment to it.

**`resource_tier_map`** — used exclusively by the dual-layer observation system. At every
decision step, the environment uses this map to:
1. Aggregate individual resource availability into per-(activity, tier) counts → the
   cluster-layer observation.
2. Translate the agent's chosen resource into a (activity, tier) label → the cluster-layer
   action.

Neither of these uses is visible to the agent. The agent only ever receives the
individual-layer observation and outputs an individual resource ID.

---

## 2. Action Space

At each decision step, exactly one activity is pending — the next task in the simulator's
event queue that requires a resource assignment. The agent's action is: **choose a specific
resource to assign to this activity.**

- **Action space:** `Discrete(n_resources)` where `n_resources` is the total number of
  unique resources in the event log.
- **Action masking:** at each step, the valid action mask is computed as the intersection of:
  - Resources that have a `resource_tier_map` entry for the pending activity (i.e. have
    empirically performed it — derived from Module 1).
  - Resources that are currently available (not busy in the simulator).
  - Invalid actions are masked out using MaskablePPO (sb3-contrib). The agent can never
    assign a resource that is ineligible or unavailable.

The effective action space at any given step is therefore only the eligible and available
resources for the pending activity — typically a small fraction of `n_resources`.

---

## 3. Observation Space

### 3.1 Individual-layer observation (given to the agent)

At each decision step, the agent receives a flat observation vector containing:

| Feature | Description | Size |
|---|---|---|
| Resource availability | Binary vector: 1 = available, 0 = busy, one entry per resource | `n_resources` |
| Queue lengths | Number of pending cases waiting at each activity | `n_activities` |
| Pending activity | One-hot encoding of the activity currently requiring assignment | `n_activities` |
| Shock flag | Binary: 1 if a shock is currently active, 0 otherwise | 1 |
| Time of day | Normalized to [0, 1] | 1 |
| Day of week | Normalized to [0, 1] | 1 |

All values are normalized to [0, 1] before concatenation. The total observation size is
`n_resources + 2 × n_activities + 3`.

### 3.2 Dual-layer observation system (cluster-layer, stored silently)

At the same decision step, in parallel, the environment computes the cluster-layer
observation using Module 1's `resource_tier_map`:

| Feature | Description | Size |
|---|---|---|
| Tier availability | For each (activity, tier) pair: count of currently available resources in that tier | `n_activities × n_tiers` |
| Queue lengths | Same as individual-layer | `n_activities` |
| Pending activity | Same as individual-layer | `n_activities` |
| Shock flag | Same as individual-layer | 1 |
| Time of day | Same as individual-layer | 1 |
| Day of week | Same as individual-layer | 1 |

After the agent selects resource `r` for the pending activity `a`, the environment
immediately looks up `resource_tier_map[r][a]` to obtain the tier label and constructs the
cluster-layer action: `(activity=a, tier=t, tier_label="slow|mid|fast")`.

The pair `(cluster_observation, cluster_action)` is appended to the **behavioral log** at
every step. This log is the direct input to Module 4. The agent's individual-layer decisions
and observations are not stored beyond what is needed for training.

---

## 4. Shock Injection

Two shock types are modelled:

**Type 1 — Resource availability shock:** a fraction `f` of the active resource pool becomes
suddenly unavailable for a duration `d`. Resources are selected uniformly at random from
those currently available. Affected resources are removed from the availability pool and
restored after duration `d`.

**Type 2 — Arrival rate spike:** the case arrival rate is multiplied by a factor `k` for a
duration `d`, generating a surge of incoming cases beyond the normal process load.

**During training:** shocks arrive according to a Poisson process (rare, independent, random
onset). Type, severity (`f` or `k`), and duration (`d`) are sampled from configurable ranges
at each episode to prevent the agent from memorising a single scenario.

**During evaluation:** a single shock of known type, severity, and timing is injected per
episode. The shock onset moment is fixed, allowing clean measurement of the before/during/
after recovery curve and computation of the resilience metrics (TI, MPD, RT, CPL).

---

## 5. Reward Functions

Two candidate reward functions are implemented. Both are computed per decision step.

### Reward Function 1 — Recovery velocity + deviation cost

Let `P(t)` be the process performance indicator (e.g. rolling average lead time of recently
completed cases) at step `t`, and `P*` the acceptable performance level calibrated from
pre-shock baseline.

```
r(t) = [Φ(s_{t+1}) − Φ(s_t)] − c · max(0, P(t) − P*)
```

- `Φ(s)` is a potential function measuring proximity to acceptable performance. Closer to
  `P*` → higher potential. This shaping term rewards step-by-step recovery velocity.
- The second term penalises every step the process spends below acceptable performance,
  proportional to the deviation magnitude.
- Hyperparameter `c` controls the trade-off between recovery speed and deviation penalty.

### Reward Function 2 — Lead time optimization with CPL constraint

```
r(t) = −lead_time_contribution(t) − λ · CPL_contribution(t)
```

- The base reward minimises average lead time per completed case.
- `CPL_contribution(t)` is the incremental cumulative performance loss at step `t`.
- `λ` is a Lagrange multiplier updated on a slower timescale than the policy:
  `λ ← max(0, λ + η · (CPL_accumulated − α · CPL_FIFO))`
  where `CPL_FIFO` is the CPL accumulated by the non-adaptive FIFO baseline under the same
  shock, and `α ∈ (0, 1)` is a configurable tolerance fraction.
- As `λ` grows, the agent is penalised more heavily for prolonged degradation beyond the
  FIFO reference. `λ` adapts automatically over training to tighten or relax the constraint.

---

## 6. Training Setup

- **Algorithm:** PPO (Proximal Policy Optimisation) via Stable-Baselines3, with action
  masking enabled through MaskablePPO (sb3-contrib).
- **Policy network:** MLP with configurable hidden layer sizes.
- **Parallelism:** multiple environment instances run in parallel during training (vectorised
  environments) for sample efficiency.
- The agent is trained until convergence on the resilience reward signal. Training episodes
  include randomly varied shocks (type, severity, timing) to produce a policy that
  generalises across shock scenarios rather than overfitting to one.

---

## 7. Behavioral Log Output (input to Module 4)

During evaluation, the trained agent's policy is frozen and run through a set of controlled
shock episodes. The behavioral log accumulated over these episodes is persisted and passed
to Module 4.

Each entry in the log corresponds to one decision step:

```json
{
  "episode_id": <int>,
  "step": <int>,
  "cluster_observation": {
    "tier_availability": {
      "<activity>_<tier_id>": <int>,
      ...
    },
    "queue_lengths": {"<activity>": <int>, ...},
    "pending_activity": "<activity>",
    "shock_active": <bool>,
    "time_of_day": <float>,
    "day_of_week": <float>
  },
  "cluster_action": {
    "activity": "<activity>",
    "tier_id": <int>,
    "tier_label": "slow | mid | fast"
  }
}
```

The individual-layer observation and the raw resource ID selected by the agent are **not**
included in this log. Module 4 operates entirely on the cluster-layer representation.

---

## 8. Open Implementation Decisions

- Exact shape of the potential function `Φ(s)` in Reward Function 1 — a linear or
  exponential decay toward `P*` are both candidates; the choice should be validated
  empirically.
- Value of `α` in Reward Function 2 (the CPL tolerance fraction relative to FIFO) — default
  candidate is 0.8, meaning the agent is penalised once its CPL exceeds 80% of FIFO's CPL.
- Whether to use a single pending-activity-at-a-time decision model or allow the agent to
  act on a batch of pending activities per step — the single-activity model is simpler and
  recommended as the default.
