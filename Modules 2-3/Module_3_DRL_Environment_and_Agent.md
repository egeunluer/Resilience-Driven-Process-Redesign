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

Two candidate reward functions are implemented. Both operate on the process performance
indicator (PPI) `φ(t)` — the running cycle time / average lead time of recently completed
cases — under the modelling assumption that **lower PPI is better**. `φ(t | eⱼ)` denotes the PPI
at time `t` under disruption `eⱼ`. Both use a consistent acceptable-performance level `P*`,
defined as the **maximum PPI observed in the event log** (e.g. max cycle time / max average
lead time) rather than a per-run pre-shock baseline, so the reference is consistent across
simulations (Kraus et al., 2024).

### Reward Function 1 — Potential-based recovery shaping + normalized deviation penalty

Built on the time-dependent resilience ratio of Henry & Ramirez-Marquez (2012), adapted so that
lower PPI is better and normalized against `P*`. A static anchor `φ_fix` stands in for the
disrupted-minimum PPI (the worst-case value, not known a priori). `φ_fix` is calibrated **before
training** — e.g. by running a set of random-policy rollouts under the shock regime and recording
the worst observed PPI; since the trained agent should beat a random policy, this is a
conservative anchor (see §8 for a more robust alternative).

Both components are normalized by the same denominator `|P* − φ_fix|`:

```
# r1 — recovery-velocity (potential-based shaping term, Ng et al. 1999):
r1(t) = [ φ(t-1 | eⱼ) − φ(t | eⱼ) ] / |P* − φ_fix|

# r2 — deviation penalty (position term):
r2(t) = min{ 0,  −( φ(t | eⱼ) − P* ) / |P* − φ_fix| }

# final reward (convex combination):
r(t)  = β · r1(t) + (1 − β) · r2(t),      β ∈ (0, 1)
```

- `r1` rewards step-by-step progress up the recovery curve (a **velocity** signal). Being the
  step change of a normalized potential, it is a potential-based shaping term and does not alter
  the optimal policy (Ng et al., 1999).
- `r2` penalizes every step spent worse than `P*`, proportional to the deviation (a **position**
  signal). The `min{0, ·}` makes it one-sided; the absolute value in the denominator keeps the
  sign correct.
- `β` trades off recovery velocity against absolute degradation.
- `r1` and `r2` share a denominator and both depend on `φ(t)` (correlated) but are complementary
  (velocity vs. position), so summing them is expected to help rather than harm (Ng et al., 1999;
  Brys et al., 2014).
- **Exact-invariance note:** the strict Ng et al. (1999) identity is `γ·Φ(s') − Φ(s)`; with a PPO
  discount `γ < 1` the exact shaping numerator is `[ φ(t-1) − γ·φ(t) ]`. The `γ` factor is dropped
  above for readability and is negligible at `γ ≈ 0.99`, but can be restored if exact policy
  invariance is required.

### Reward Function 2 — Constrained-MDP average lead time with CPL as a shock-period SLA

Minimizes average lead time while enforcing a Cumulative-Performance-Loss (CPL) budget as a
constraint. Because the agent decides at discrete epochs while the process flows in continuous
time, the environment is treated as a Semi-Markov Decision Process (SMDP), so continuous holding
costs accumulate correctly across variable-length epochs (Bradtke & Duff, 1995; Puterman, 1994;
Gosavi, 2015).

Definitions:
- `d(t) = max{0, φ(t | eⱼ) − P*}` — instantaneous deviation from `P*`.
- `τ_k = t_k − t_{k-1}` — SMDP sojourn time between consecutive decision epochs (requires the
  continuous decision-epoch time `t_k` exposed by Module 2 — see Module 2 §4).
- `B_k = B_{k-1} + d(k) · τ_k`, `B_0 = 0` — accumulated budget utilization; equals the resilience
  loss `∫ d(t) dt` (Bruneau et al., 2003).
- `CPL_base` — total CPL of a non-adaptive baseline (e.g. FIFO) under the same shock.
- `CPL_max = p · CPL_base`, `p ∈ (0, 1]` — the budget; e.g. `p = 0.5` starts penalizing once the
  agent exceeds 50% of FIFO's CPL (Ray et al., 2019).
- `u_k = B_k / CPL_max` — normalized budget utilization.
- `v_k = max{0, u_k − 1}` — constraint-violation signal.

```
# reward: Meneghello et al. (2024) base term (gated by a completion flag)
#         minus a Lagrangian violation penalty:
r(k) = f1 · ( 1 / [ 1 + c(σ) / C ] )  −  f2 · λ_k · v_k

# Lagrange multiplier, updated on a slower timescale (RCPO, Tessler et al. 2018):
λ_{k+1} = max{ 0,  λ_k + η_λ · v_k },      η_λ ≪ η_θ
```

- `c(σ)` — lead time of a completed trace `σ`; `C` — scaling constant (Meneghello et al. 2024
  used 10000; must be tuned per process / time unit).
- `f1` — 1 when a trace completes at this step (augments the Meneghello reward so it is not only
  granted after actions); `f2` — 1 when the agent takes an action.
- `λ_k` grows while the CPL budget is violated (`v_k > 0`) and relaxes otherwise, adapting the
  constraint automatically; `η_λ` must be far smaller than the policy learning rate `η_θ` for
  stable multi-timescale convergence (Tessler et al., 2018).
- **Observation dependency:** if RF2 is used, the individual-layer observation (§3.1) must be
  augmented with `u_k` so the agent can distinguish early vs. late shock periods and learn a
  budget-aware policy.

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

- Calibration of `φ_fix` in Reward Function 1 (the disrupted-minimum anchor). Default: the
  worst PPI over a set of random-policy rollouts. A robust statistic (e.g. a high percentile of
  the disrupted-window PPI) or the trough of a heuristic baseline (e.g. FIFO, consistent with
  RF2's `CPL_base`) is preferable to the single worst value, which is outlier- and
  sample-size-sensitive. Also fix `β ∈ (0, 1)`. Since `φ_fix` enters only through the constant
  `|P* − φ_fix|`, its main effect is reward scaling — verify it keeps rewards O(1) and never near
  zero.
- Value of `p ∈ (0, 1]` in Reward Function 2 (the CPL budget as a fraction of FIFO's `CPL_base`)
  — e.g. `p = 0.5` penalizes once the agent exceeds 50% of FIFO's CPL. Also tune the scaling
  constant `C` (Meneghello et al. 2024 used 10000) and ensure `η_λ ≪ η_θ`.
- Whether to use a single pending-activity-at-a-time decision model or allow the agent to
  act on a batch of pending activities per step — the single-activity model is simpler and
  recommended as the default.
