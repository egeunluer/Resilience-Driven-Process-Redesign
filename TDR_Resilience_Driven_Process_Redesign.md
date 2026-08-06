# Thesis Description Report
## Resilience-Driven Process Redesign: Deep Reinforcement Learning for Shock Recovery via Augmented Process Discovery

**Author:** Ege Ünlüer  
**Matrikelnummer:** 03790918  
**Contact:** ege.unluer18@gmail.com | go72tuw@tum.de

---

## 1. The Core Problem

Business processes are typically designed and optimised for stable, predictable conditions. When something unexpected hits — a key team suddenly loses a third of its staff, or an external event causes a sudden surge in incoming cases — organisations have no systematic playbook for how to reallocate their resources to absorb the disruption and recover quickly. The BPM literature acknowledges this gap but offers no concrete, automated solution.

At the same time, Deep Reinforcement Learning (DRL) has shown it can learn powerful resource allocation policies for business processes. The catch: the policies it learns are black boxes. A process manager cannot read, audit, or deploy a neural network as a redesign artefact. They need something they can look at, question, and hand to a team.

This thesis closes both gaps simultaneously: it trains a DRL agent to learn *how* to keep a process resilient under shocks, then translates what the agent learned into a human-readable process redesign that a manager can actually use.

---

## 2. What the Thesis Produces

The central deliverable is a **White-Box Resilience Engine**: a pipeline that takes an event log as input and produces, as output, a formal process redesign artefact — either an augmented BPMN diagram with XOR gateways and human-readable branching conditions, or a BPMN model paired with DMN decision tables that spell out allocation rules per task. Both output formats are readable, auditable, and deployable without the trained neural network being present.

---

## 3. Research Questions

**Main RQ:** What resource allocation strategies does a DRL agent learn to keep a business process resilient under exogenous shocks, and what faithful, human-readable process redesign do those strategies distil into?

- **RQ1:** How are shock and resilience concepts defined in Business Process Management literature and how can they be integrated into process simulations?
- **RQ2:** How can the DRL agent learn to increase resilience of business processes in shock states and how can the performance of the agent be evaluated?
- **RQ3:** How can the decision-making of the trained DRL agent be transformed into an admissible business process model, and does the generated process model accurately reflect the agent’s decisions?
- **RQ4:** Are the resilience gains achieved by the DRL policy and the fidelity with which the redesign artifact represents that policy independent of one another, and under which combinations of the two can the White-Box Resilience Engine be considered valid?
---

## 4. Methodology Overview

The methodology is structured as four sequential modules. Each module runs independently and passes a well-defined artefact to the next. The whole pipeline is applied separately for each event log.

### Module 1 — Empirical Pre-Processing

Before anything else, the raw event log is processed to extract two empirical artifacts that all downstream modules depend on.

**Stage 1 — Processing time computation.** For every (resource, activity) pair observed in the log, the median elapsed time between the `start` and `complete` lifecycle events is computed. Where only single timestamps exist per event (no start/complete pair), processing time is approximated as the time delta to the previous event in the same case — a sojourn time approximation that includes waiting time, not just active working time. Before computing the median, values above the 95th percentile of each (resource, activity) pair are capped to remove outliers. Any pair with a zero or negative value is excluded and logged as a data anomaly. The result is a `resource_processing_times` map, consumed by Module 2 for duration sampling when the agent assigns a specific resource to a specific activity.

**Stage 2 — Activity-specific performance tiering.** For each activity independently, the resources that have performed it are clustered into performance tiers using quantile binning — resources are divided into equal-sized groups ranked by their processing time for that activity. The default is three tiers (0 = slow, 1 = mid, 2 = fast), clamped to the number of eligible resources when fewer are available. The result is a `resource_tier_map` that records, for every resource, its tier label per activity it has performed. Resources that have never performed an activity receive no entry for it — they are simply ineligible for assignment to it.

The `resource_tier_map` is never exposed to the DRL agent. It is used exclusively by Module 3's dual-layer observation system (to maintain a silent cluster-layer view alongside the agent's individual-layer decisions) and by Module 4's distillation pipeline (to translate individual resource choices into tier-level representations that interpretable models can operate on).

### Module 2 — The Hybrid Simulation Model

The simulation backbone is RIMS (Runtime Integration of Machine Learning and Simulation), an established state-of-the-art hybrid simulator that uses ML predictions for activity durations and routing rather than fixed distributions. This thesis extends RIMS in two ways.

First, RIMS's duration model is replaced with the empirical `resource_processing_times` from Module 1, so the simulator samples the duration of an activity from the historical processing time of the specific resource assigned to it. Resources with no entry for a given activity have never performed it and are ineligible for assignment.

Second, a **shock injection mechanism** is added. Two types of shocks are modelled, following definitions in the literature: a fraction of the active resource pool becoming suddenly unavailable, and a sudden spike in the rate of incoming cases. Shocks arrive according to a Poisson process (rare, independent, random onset), persist for a defined duration, and vary in type, size, and timing during training to prevent the agent from memorising a single scenario. At evaluation time, shocks are controlled precisely — one shock per run, at a known moment — so that a clean before/during/after recovery curve can be measured.

### Module 3 — The DRL Environment and Agent

The simulator from Module 2 is wrapped in a standard Gymnasium interface, turning it into a reinforcement learning environment. The agent operates entirely at the **individual resource level** — each action is assigning a specific resource to the pending activity.

**Individual-layer observation (given to the agent).** At each decision step, the agent receives a flat observation vector containing: per-resource availability (binary, one entry per resource), queue lengths at each activity, a one-hot encoding of the pending activity requiring assignment, a binary shock flag indicating whether a shock is currently active, and normalised time-of-day and day-of-week context. The agent never sees or reasons about performance tiers.

**Action masking.** At each step, the valid action mask is the intersection of resources that have a `resource_tier_map` entry for the pending activity (i.e. have empirically performed it) and resources that are currently available. The agent can never assign a resource that is ineligible or unavailable, which improves learning efficiency. Action masking is implemented via MaskablePPO (sb3-contrib).

**Dual-layer observation system (cluster-layer, stored silently).** In parallel with the individual-layer, the environment uses Module 1's `resource_tier_map` at every step to compute a cluster-layer observation: for each (activity, tier) pair, the count of currently available resources in that tier. After the agent selects a specific resource for the pending activity, the environment immediately translates this into a tier-level action by looking up `resource_tier_map[resource][activity]`. The pair (cluster-layer observation, cluster-layer action) is appended to the **behavioral log** at every step. The agent is entirely unaware of this parallel layer — it only ever receives the individual-layer observation and outputs a specific resource ID.

The reward function is the most critical engineering component of this module. Two candidate designs are developed:

**Reward Function 1** combines a potential-based shaping term that rewards step-by-step progress along the recovery curve (recovery velocity) with a penalty for every timestep the process spends performing worse than its acceptable performance level (deviation cost). The two signals are complementary — one tells the agent how fast it is recovering, the other tells it how far below acceptable it currently sits — and together they push the agent to minimise the depth and duration of performance loss.

**Reward Function 2** takes a constrained optimisation approach. The base reward optimises average lead time (the same signal as prior work), but a CPL (Cumulative Performance Loss) budget is enforced as a constraint using a Lagrange multiplier that is updated on a slower timescale than the policy. This naturally penalises the agent for allowing prolonged degradation beyond a threshold calibrated against a non-adaptive baseline (FIFO), and the multiplier adapts over training to tighten or relax the constraint automatically.

The agent is trained with the PPO (Proximal Policy Optimisation) algorithm via Stable-Baselines3.

### Module 4 — Policy Distillation into a Process Redesign Artefact

Once trained, the agent's policy is frozen and used to generate a behavioral dataset by running it through many shock-injected simulation episodes. At every decision step, the environment records the cluster-layer observation (tier availability counts per activity, queue lengths, pending activity, shock flag, time context) and the cluster-layer action (which performance tier the assigned resource belongs to, for the pending activity). These (cluster_observation, cluster_action) pairs form the behavioral log that Module 4 consumes. The agent's individual-layer observations and raw resource selections are not stored — Module 4 operates entirely on the tier-level representation.

Two distillation approaches are proposed, representing the two output format choices:

**Approach A — Tier-Enriched Process Discovery (original):** Each logged event's activity label is rewritten to include the performance tier selected (e.g. "Assess Fraud [fast-tier]" vs "Assess Fraud [slow-tier]"). This makes resource allocation decisions visible as control-flow choices. The enriched log is then mined with the Inductive Miner (IMf variant), which produces a sound, block-structured process model. Where the agent routes the same base activity to different tiers depending on conditions, the miner surfaces XOR gateways. Decision mining then recovers the branching logic at each gateway — fitting interpretable decision trees to the cluster-layer observation features — and translates them into human-readable guard conditions (e.g. "if shock is active and fast-tier availability for this activity drops to zero, route to mid-tier"). The result is an augmented BPMN diagram that encodes not just what happens but which performance tier handles it and under what circumstances.

**Approach B — DMN-Based Distillation (alternative):** Rather than encoding tier decisions in the control flow, this approach keeps the base BPMN control flow unchanged and attaches a per-activity DMN (Decision Model and Notation) decision table to each activity. Each table is trained as an interpretable decision tree that predicts which performance tier the agent would assign, given the current cluster-layer observation state. Every path through the tree becomes one rule in the DMN table — auditable, editable, and executable without the agent. The full artefact is a BPMN + DMN model that separates routing logic (BPMN) from resource allocation logic (DMN), which is architecturally cleaner and standard-compliant.

Both approaches include a fidelity evaluation step: the distilled artefact is assessed on a held-out portion of the behavioral dataset for fitness (does it cover the agent's real decisions?) and precision (does it allow only what the agent actually did?). The decision-tree branch conditions are also evaluated for predictive accuracy against a majority-class baseline. Finally, the distilled artefact is run as a fourth decision-maker alongside the trained agent and the three baselines on the same shock scenarios, and its resilience metrics are compared directly against all of them.

---

## 5. Resilience Definition and Metrics

Resilience is understood here as having three components: absorbing the impact of a disruption, adapting to changed conditions during it, and recovering performance afterwards. Both preparedness (how quickly the system is affected and by how much) and recovery speed matter.

The metrics used to measure this come from Kraus et al. (2024):

- **Time to Impact (TI):** how long after a shock onset before performance degrades — a preparedness indicator.
- **Maximum Performance Deviation (MPD):** how far performance drops at its worst — also a preparedness indicator.
- **Recovery Time (RT):** how long it takes to return to acceptable performance — a recovery indicator.
- **Cumulative Performance Loss (CPL):** the total area of degradation below the acceptable performance level across the full shock episode — captures both depth and duration of disruption.

These metrics can be read directly from a graph of the process performance indicator (e.g. average lead time) over time, structured around the disruption event. Which metrics the agent directly optimises depends on the reward function chosen; the others serve as independent evaluation metrics.

---

## 6. Evaluation Design

Every decision-maker — the trained DRL agent, and three baselines — is evaluated on the same set of controlled, single-shock simulation runs.

**Baseline 1** is a DRL agent with the same architecture but trained only on the steady-state reward (no shock flag, no resilience-oriented reward), representing the prior state of the art. Comparing against it isolates the contribution of the resilience-specific reward design.

**Baseline 2** is a shortest-queue heuristic that always assigns the pending activity to whichever eligible resource has the most available capacity. It is a sensible, reactive rule, and beating it demonstrates that learning adds genuine value over simple reactive logic.

**Baseline 3** is a non-adaptive FIFO rule, static and shock-oblivious. It also defines the CPL budget used in Reward Function 2 — the agent is penalised once its CPL exceeds a fraction of what FIFO accumulates under the same shock.

Four experiments structure the evaluation:

1. **Resilience benchmark (main):** agent vs. all three baselines on the full resilience metric set.
2. **Shock-aware reward ablation:** agent vs. Baseline 1 only, isolating the reward engineering contribution.
3. **Anticipation vs. reaction:** agent vs. Baseline 2, testing whether seeing the shock flag in advance produces better preparedness than purely reactive queue-based routing.
4. **Steady-state cost of resilience:** all four decision-makers compared in a no-shock control condition, measuring any efficiency penalty the resilience-trained agent pays during normal operations.

---

## 7. Position in the Literature

The thesis sits within prescriptive process monitoring — the family of methods that observe running processes and recommend runtime interventions. It reorients the field's usual objective (optimising efficiency or deadline compliance) toward resilience under exogenous shocks.

It differs from prior prescriptive monitoring work in two ways. First, the intervention is not prediction-based; the agent learns a policy directly through reward maximisation rather than first predicting outcomes and then choosing interventions. Second, the learned resource-allocation intervention is subsequently distilled into a control-flow artefact (BPMN ± DMN), so the output simultaneously represents a runtime prescriptive system and a static process redesign — bridging two streams of literature that are usually treated separately.

---

## 8. Target Event Logs

The methodology is validated on four real-world event logs, covering different process domains and resource structures:

| Log | Domain | Resource structure |
|---|---|---|
| BPI12W | Loan application processing | Individual workers |
| BPI2017 | Loan applications (larger, cleaner) | Individual workers |
| Sepsis Cases | Hospital patient pathways | Departments (not individuals) |
| BPI2020 | Travel reimbursement at TU Eindhoven | Individual workers, multiple approval roles |

**Dataset-specific notes:**

- **BPI12W:** Only `W_`-prefixed activities are processed — automated `A_` and `O_` events do not represent human resource decisions. Only `COMPLETE` lifecycle events are present, so the sojourn approximation applies for processing time computation.
- **BPI2017:** Only `W_`-prefixed activities are processed. Has `start`/`complete` lifecycle pairs, so true processing times are available.
- **Sepsis Cases:** Resources are recorded as departments rather than individual workers. Department identifiers are treated as resource IDs throughout — no structural adaptation is required in Module 1, since processing time computation and activity-specific tiering apply equally to department-level resources.
