# A white-box resilience engine for business processes: approach description

Status: 2026-11-06. This document describes the approach: what is modelled, measured,
learned and compared, and why. The implementation history and run logs are in
`New_Approach_Implementation_Plan.md`; every parameter with its source is in
`Modules 2-3/Assumptions_Table.md`.

---

## 1. The approach in brief

A business process runs normally until a **shock** hits it, for example an IT outage
or a surge of incomplete applications. Some cases are then blocked or pushed into
extra work, and if nothing is done, many of them miss their deadlines. Organisations
handle this with **recovery actions**: wait, retry, reassign, use an alternative
channel, or escalate to a person. Choosing well is hard:

- the shock's length is unknown when the decision is made;
- the stronger actions use scarce people or have side effects;
- the consequences show up days or weeks later, often on other cases.

The approach has four parts:

1. **A validated simulation model** of a real loan-application process (BPIC 2017),
   mined from the full event log, into which shocks are injected.
2. **A resilience measure** of the whole process, Q(t), and the area of its dip, the
   Loss (the resilience triangle).
3. **A deep reinforcement learning (DRL) agent** that learns, for every case a shock
   affects, which recovery action to take, so that the process loses as little
   resilience as possible at an acceptable cost.
4. **Distillation** of the learned policy into a **DMN decision table embedded in
   BPMN**: the *white-box resilience engine*. The black-box policy becomes an explicit,
   inspectable and executable process-model artifact.

The agent decides *how a disrupted case is recovered*. It does not allocate resources
in normal operation. An earlier allocation agent learned to pick the fastest
resource, which is efficiency, not resilience.

---

## 2. The process and its simulation model

### 2.1 Data
- **Log:** BPI Challenge 2017, the full log with application (A_), offer (O_) and
  workflow (W_) events. 31,509 loan applications, about 86 new applications per
  day, about 1,900 open at any time.
- **Process facts the model reproduces:**
  - median cycle time 19 days, mostly spent waiting for the customer;
  - offers expire automatically 30 days after being sent;
  - 48% of cases go through a missing-documents loop;
  - 65% of applications arrive online;
  - customer work items (W_) take minutes per session.

### 2.2 Stage model
The process is modelled as **stages**, mined from the log:

| Stage | What happens |
|---|---|
| intake | the application is created (and leads handled for online ones) |
| complete | the application is completed and an offer is prepared |
| await_customer | the offer is out; the customer returns it, or it expires after 30 days |
| validation | the returned documents are checked |
| incomplete | documents are missing; the customer is called and uploads them |

It ends in one of three outcomes: accepted, denied or cancelled.

Mined from the log, with no assumptions:
- arrivals per hour of the week;
- routing probabilities per stage (by visit number and by flags such as online
  origin or fraud check);
- customer waiting times as empirical distributions;
- the 30-day offer expiry as a race between the customer and the timer;
- offers per case;
- the work items per stage and the number of work sessions per visit;
- per worker and work item, qualification and session-duration distributions;
- worker calendars: Monday–Saturday 07:00–21:00, with each worker's daily attendance
  probability mined from the log.

### 2.3 Resources
- **Constrained pool:** for each work item, the 3 most active workers in the log,
  16 workers in total.
- **Fixed dispatcher:** each work session goes to the fastest free qualified worker,
  otherwise into a first-in-first-out queue per work item. The dispatcher never
  changes; the agent does not allocate resources.
- **Escalation backup team:** for each work item that can be escalated, the 2 workers
  ranked just outside the pool. They are real workers of the log (qualification and
  attendance mined) who only take escalations. They are scarce.

### 2.4 Validation
Compared with the real log (3 seeds × 240 days), the simulator passes all 9 checks:
- cycle-time distribution (KS distance 0.059; median 20.8 vs 19.1 days);
- outcome shares (accepted 53.4% vs 54.7%, cancelled 34.7% vs 33.1%, denied 11.9% vs 11.9%);
- incomplete-loop share (46.5% vs 47.6%);
- open cases (median 1,908 vs 1,937);
- arrivals per day (85.3 vs 86.2).

### 2.5 Common random numbers
Every source of randomness has its own random stream:
- arrivals;
- worker attendance;
- the backup team's attendance;
- per case: one stream per stage visit for routing and waits, one per stage visit for
  work durations, and one for the outcomes of its recovery actions.

An action taken on one case therefore changes no other case's random draws. Two runs
of the same episode that differ only in the actions are the same world, case by case.
This makes paired comparisons precise, and it makes the *do-nothing twin* (§5.3)
possible.

---

## 3. Measuring resilience

### 3.1 On track and Q(t)
- Each stage k has a **deadline** D_k: the 90th percentile of case age when leaving
  that stage in the log.

  | Stage | D_k |
  |---|---|
  | intake | 1.0 d |
  | complete | 3.6 d |
  | await_customer | 32.8 d |
  | validation | 28.4 d |
  | incomplete | 35.5 d |

- A case is **on track** while its age is at most D_k of the stage it is in.
- **Q(t)** = the share of open cases that are on track. It is a system-level service
  level in [0, 1]. Normal operation runs at about 0.90; a shock pushes it down.

### 3.2 Loss: the resilience triangle
Loss = ∫ (1 − Q(t)) dt over the 50 days after the shock's onset.

- **No threshold.** Every moment any case is late counts. An earlier version only
  counted Q below a "bad normal day" level (Q_ref). That made about half of all
  deadline misses free and hid the benefit of acting.
- **Unit: Q-day.** 1 Q-day = Q at 0 for one day ≈ 1,900 case-days of lateness.
- **Normal operation** alone accrues about 4.64 Q-days per 50 days. A D1–D3 shock adds
  0.11–0.18 Q-days on average (Loss_base, §5.5).
- **It is integrated exactly:** Q is piecewise constant, and deadline crossings are
  processed in time order.

### 3.3 Reported alongside
- lowest 24-hour rolling Q (robustness);
- time to recover (the shock is over, every blocked step is done, and the 24-hour
  rolling Q is back at its normal mean);
- offer expiries caused by delays;
- action counts and action cost.

### 3.4 Episodes
- **Warm-up:** 40 simulated days, so the process is in its normal state.
- **Onset:** the shock starts at a random moment of working time in the following
  week.
- **Horizon:** 50 days after the onset, fixed. Delays often turn into missed deadlines
  only weeks later; most of a D3 shock's harm arrives at days 25–45.

---

## 4. Shocks, decisions and recovery actions

### 4.1 Shock types (training set: D1–D3)

| ID | Shock | Mechanism | Length |
|---|---|---|---|
| D1 | Offer dispatch outage | Cases finishing `complete` cannot have their offer sent (`complete → await_customer` blocked) | 72–120 h, uniform, hidden from the agent |
| D2 | Telephony outage | The calls `W_Call after offers` and `W_Call incomplete files` cannot start | 72–120 h, hidden |
| D3 | Missing-documents surge | A document-portal fault: the probability that validation finds a file incomplete rises by 50%, and the customer's upload is held up 72–168 h | 72–120 h, hidden |
| D4 | Demand spike (×3 arrivals) | Defined but **not trained**: under the threshold-free Loss it adds 0.03 ± 0.21 Q-days, no measurable harm | — |

Shock types follow the shock taxonomy of the literature review (unavailability,
exogenous shocks, exceptions, failures). Their failure types are grounded in the
corpus: system failures and external service disruptions (P20), communication
failures (rBPMN, P36–P40).

### 4.2 When a decision is made
- **D1, D2:** a case reaches a blocked step.
- **D3:** a case is routed into the missing-documents loop during the surge.
- **SLA-risk alert:** a case left waiting gets a second decision when its slack
  (time until its stage deadline) falls to 48 hours.

There are about 280–890 decisions per episode, depending on the shock.

### 4.3 Recovery actions
The repertoire follows the exception-handling patterns of the literature (Russell,
van der Aalst & ter Hofstede 2006; Lerner et al. 2010; corpus P20).

| Action | Effect | Available for | Cost κ |
|---|---|---|---|
| do_nothing | wait for the service to return or the loop to run its course | all | 0 |
| retry | automatic document re-fetch after 1 working hour; finds the documents with p = 0.4, so the case skips the loop; at most 2 per case | D3 | 0.05 |
| reassign | a caseworker chases the documents directly (no portal delay) and the case's calls go to the front of the queue | D3 | 0.15 |
| substitute | the blocked step is done another way and the case rejoins its normal path. D1: the offer goes through the online-only channel (exists in the log: `O_Sent (online only)`, 4.9% of offers); 20% of customers miss it and it is re-sent 24–72 h later. D2: written contact instead of a call; the customer answers 24–72 h later | D1, D2 | 0.20 |
| escalate | a backup worker does the step by hand: 0.5–2 working hours, success 0.95. Masked while no backup worker for that step is on shift and free | D1, D2, D3 | 0.30 |

κ values are relative costs in units of c_p, the average do-nothing harm per decision.
Their order (retry < reassign < substitute < escalate) follows the cost dimensions of
P20: manual effort, implementation effort, risk of later errors. The absolute values
are assumptions and are swept in the experiments.

---

## 5. Learning: reward, objective and agent

### 5.1 Decisions per case
Each decision is an agent step. The decisions of one case (on average 1.15–1.2) form
a short chain, and each chain is learned from as its own short episode. The agent
therefore learns, for a case in a given situation, which action gives the best
outcome.

### 5.2 Reward of a decision
For decision i of case c:

  r_i = −[ κ(a_i) + shadow(a_i) + (N_e / Loss_base,e) · ( (1−β)·ΔL_c,i + β·S_i ) ]

- **Own part, ΔL_c,i:** case c's share of the Loss from this decision until its next
  one (or the horizon). At every moment the Loss rate (1 − Q) is split equally among
  the off-track cases, so each late case accrues dt/open. A case whose offer expires
  keeps being charged after the expiry.
- **Shared part, S_i:** the Loss of the whole process at each moment is split equally
  among all decisions made before it. This carries effects on other cases, such as
  the queues relieved by avoided document loops in D3. It is bounded: no single
  decision can receive a large share.
- **Weight β = 0.5** between the two parts.
- **N_e / Loss_base,e:** puts both Loss parts in per-decision units, the units of κ.
- **The Loss parts arrive at the end of the episode;** the action cost at the step.

### 5.3 The do-nothing twin
At the end of every episode, the same episode is replayed with every decision
answered do_nothing. Thanks to the separate random streams it is the same world.
Both Loss parts count only the **difference to the twin**:

- a case's own lateness minus its lateness in the twin over the same interval;
- the process Loss minus the twin's.

Lateness that would have happened anyway, in normal operation or with the shock left
alone, cancels case by case. What remains is what the actions changed. The twin does
not depend on the agent's actions, so subtracting it does not change the best
policy; it removes noise. An agent that always does nothing gets exactly 0 for both
Loss parts.

### 5.4 Shadow price of the backup workers
The per-case reward cannot see that an escalation keeps a scarce backup worker from
the next case: that harm falls on another case. Without a correction, the agent
escalated every case it could in D3, including cases with weeks of slack.

The **shadow price** is the value of one escalation's worker time to the rest of the
process. It is measured from the process, under the current policy:
- run the same episodes with 2 and with 1 backup worker per step;
- divide the extra Loss by the escalations forgone.

Results, measured under an agent trained without a price: D1 0 (it hardly
escalates), D2 0 (no measurable cost), D3 1.19 per escalation. These values are kept
fixed. A re-measurement under the agent trained with them gave D2 1.34 and D3 3.54:
once escalations go to urgent cases, each one is worth more. It is reported as a
check, not used. Every escalation of a team is charged (1−β) × its price. The charge never
looks at the case's own urgency, so the agent still has to learn when an escalation
is worth it. It is part of the reward only; the objective holds the real cost in the
Loss. It depends on the policy that uses the workers (see the re-measurement above).

### 5.5 Episode objective (evaluation and benchmark)
  objective = 1 + (Loss − Loss_do-nothing) / Loss_base,e + Σκ / N_e

- **Loss_do-nothing:** the Loss of the same episode under do_nothing (the twin).
- **Loss_base,e:** the mean extra Loss a shock of type e causes under do_nothing,
  measured against the same episode without the shock: D1 0.178, D2 0.152,
  D3 0.110 Q-days.
- **N_e:** decisions per episode under do_nothing: D1 648, D2 890, D3 286.
- **Reading it:** do_nothing = 1 exactly; 0 = the shock did no harm at no cost; below
  0 = less lateness than without the shock (the actions also spared cases loops they
  would have gone through anyway).

### 5.6 What the agent observes (38 features, all in [−1, 1])
- **Shock:** active; time since onset; time since end; scenario (one-hot); cause
  (one-hot); SLA-alert flag.
- **Time:** working hours now; hour of day.
- **Capacity and load:** backup workers free for this step; decisions pending; cases
  blocked; blocked cases with slack < 2 days and 2–7 days; open cases; arrivals in the
  last 24 h; work waiting for a worker.
- **Process state:** Q; 24-hour rolling Q.
- **The case:** its stage (one-hot); slack to its stage deadline; age; time to its
  offer expiry; attempts so far; retries left; visit number; offers; fraud flag; demand
  spike cohort flag.

The shock's remaining length is hidden.

### 5.7 Algorithm
- **MaskablePPO** (invalid actions masked), adapted for per-case chains
  (`CaseMaskablePPO`). Whole episodes are collected, steps are regrouped into case
  chains when each episode ends, and values and log-probabilities are recomputed
  before GAE and the PPO update.
- **Settings:**
  - γ = 1, GAE λ = 0.95;
  - learning rate 3·10⁻⁴, constant;
  - entropy bonus 0.01;
  - 6 parallel environments, 1,024 steps each per update, minibatch 256;
  - networks with 2 × 64 units (policy and value).
- **Training cycle:** D1, D2, D3 in turn, shock lengths drawn per episode.
- **Evaluation:** every 5 updates, 24 fixed episodes (seeds 1000+, 8 per scenario,
  deterministic policy). Every evaluated model is kept. The final model is chosen on
  the separate benchmark set, not on the evaluation set.

---

## 6. Baselines and evaluation protocol

### 6.1 Baseline ladder (from weak to strong)
1. **do_nothing:** no adaptation; the lower bound.
2. **Single-action policies:** always retry / reassign / substitute / escalate (or
   do_nothing where not allowed). Tests whether one action dominates.
3. **Greedy:** for each decision, estimates every allowed action's delay for this case
   (an outage's expected remaining length given how long it has lasted, the side
   effects of substitute, the escalation handling time, the loop length in D3). It
   takes the cheapest action that keeps the case within its deadline. Myopic, with
   no long-term or cross-case view.
4. **Expert rule** (hand-written, in the style of P20 patterns and P11/P17 rules), with
   urgent = slack < 7 days:
   - **D3:** urgent → escalate, else reassign; not urgent → retry;
   - **outage:** urgent → escalate, else substitute; not urgent → wait;
   - **SLA alert:** escalate, else substitute.

   The threshold is also swept (1–21 days) to report the best-tuned rule.
5. **The DRL agent.**
6. **Later:** a Metzger-style single-action trigger agent, and an upper bound on a
   small instance.

### 6.2 Protocol
- **Benchmark episodes:** seeds 200+, separate from training and evaluation; 16 per
  scenario.
- **Pairing:** all policies run on the same episodes. Results are reported as paired
  differences with 95% intervals, against do-nothing and against the rule.

---

## 7. Results so far (benchmark, 16 episodes per scenario)

| Policy | Objective | D1 | D2 | D3 |
|---|---|---|---|---|
| do_nothing | 1.000 | 1.000 | 1.000 | 1.000 |
| always retry | 0.758 | 1.000 | 1.000 | 0.273 |
| always reassign | 0.923 | 1.000 | 1.000 | 0.769 |
| always substitute | 0.772 | 0.242 | 1.075 | 1.000 |
| always escalate | 0.872 | 0.919 | 0.980 | 0.718 |
| greedy | 0.785 | 0.572 | 0.993 | 0.789 |
| rule (7 d) | 0.371 | 0.257 | 0.965 | −0.111 |
| rule, best threshold (1 d) | 0.359 | 0.237 | 0.970 | −0.130 |
| DRL agent, no shadow price (61k–200k) | 0.394–0.398 | 0.23–0.25 | 0.79–0.80 | 0.149 |
| **DRL agent, shadow price, fresh (100k)** | **0.303** | **0.226** | **0.792** | **−0.108** |

- **Agent (fresh, shadow price) vs rule, same episodes:**
  - D1: −0.03 ± 0.04 (tie);
  - **D2: −0.17 ± 0.10 (agent better)**;
  - D3: +0.00 ± 0.10 (tie);
  - **overall: −0.067 ± 0.051 (agent better)**.
- **Against the best-tuned rule:** overall −0.055 ± 0.048 (agent better).
- **Without the shadow price,** the agent tied the rule overall and lost D3
  (+0.26 ± 0.12).
- **What the agent learned:**
  - **D1:** waits first and substitutes as cases approach their deadline; the share
    substituted rises from 26% to 59% as slack shrinks.
  - **D2:** found which cases benefit from written contact, where no rule helps.
  - **D3:** retries most cases and escalates by urgency: 99% of overdue cases when a
    backup worker is free, falling to 0% beyond 14 days of slack. Without the shadow
    price it escalated every case it could (§5.4).

---

## 8. From policy to white-box resilience engine

1. **Roll out** the trained policy on many episodes and record (observation, action)
   per decision.
2. **Fit a shallow decision tree per failure type** (depth 3–5) that maps the
   observation to the action.
3. **Measure fidelity both ways:**
   - **model → agent:** the share of decisions where tree and agent agree (target
     ≥ 90%);
   - **agent → model:** run the tree itself as the policy on the benchmark and compare
     its resilience with the agent's.
4. **Export DMN 1.3:** one decision table per failure type, with rows like "if slack
   < x and backup worker free → escalate".
5. **Embed in BPMN:** a boundary error event on each step a shock can block (or a
   conditional event for the alert) leads to a business-rule task that calls the DMN
   table, followed by the chosen recovery path.
6. **Execute and evaluate the engine:** the BPMN + DMN artifact is run as the policy
   in the simulator and benchmarked like every other policy.

---

## 9. Research questions and how they are answered

| RQ | Question | Operationalised by |
|---|---|---|
| RQ1 | How can the DRL agent learn to increase the resilience of business processes in shock states, and does the agent enhance resilience after training? | The setup of §2–§5. Benchmark objective, Loss and recovery against do-nothing, single actions, greedy and the (tuned) rule, as paired differences with 95% intervals per shock type |
| RQ2 | How can the decision-making of the trained DRL agent be transformed into an explainable and interpretable business process model, and does the generated process model accurately reflect the agent's decisions while the agent itself accurately mirrors the process model? | The distillation of §8. Fidelity (agreement rate per failure type); the performance gap between tree policy and agent; table size and depth as interpretability |
| RQ3 | Are the resilience gains from the DRL policy and the accuracy of the BPMN + DMN artifact independent of each other? And does the promised white-box resilience engine increase resilience while also successfully integrating the agent's policy into the process model? | The relationship between tree depth (simplicity), fidelity and resilience across distillation settings; the executed engine benchmarked against do-nothing, rule and agent |

---

## 10. Assumptions and limitations

**Assumed rather than mined** (all in the assumptions table, the key ones swept):
- shock mechanisms and lengths;
- action effects: re-fetch success, re-send rate, written-contact delay, escalation
  handling time and success;
- κ values and β;
- the size of the backup team;
- the 48-hour SLA alert;
- the rule's threshold.

**Limitations:**
- **The process is modelled at stage level:** waits and routing are mined, but
  actions only change what the model represents.
- **Shocks are one at a time;** combined shocks are a planned out-of-distribution
  test.
- **The shadow price depends on the policy** and is re-measured after training.
- **Per-case decisions cannot see every cross-case effect.** The shared reward term
  and the shadow price cover the known ones (relieved queues, scarce backup workers).
- **The results come from one simulated process** (BPIC 2017). Generalisation needs a
  second log.

**Alternative kept in reserve:** a process-level agent that switches redesign
patterns (standard / triage / alternative channel / reserve team) under an outage of
unknown length, with a budget instead of prices
(`Simplified_Design_Process_Level_Agent.md`).
