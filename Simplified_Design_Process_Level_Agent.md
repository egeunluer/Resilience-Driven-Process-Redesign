# Simplified design: a process-level redesign agent

Status: proposal, 2026-11-05. Alternative to the per-case recovery agent of Phase 9.

## 1. Why simplify

The per-case agent makes one decision per affected case. A disruption episode has
250–900 such decisions, and they interact through shared workers, queues and Q(t).
Almost every assumption added in Phase 9 follows from that choice:

| Source | What it forced |
|---|---|
| One decision per case, effects on other cases weeks later | per-case Loss shares, β, the sharing window, `CaseMaskablePPO`, the twin |
| A price per action | five κ values that the log cannot ground; grounding them made single actions dominant |
| Four disruption types with their own action semantics | substitute, escalate etc. defined, costed and given side effects per type |

The per-case agent is also operational control, not redesign: its output says what to
do with one case, not how the process should run under a disruption.

## 2. The idea in one sentence

During a disruption, the agent decides every half working day **which process
variant runs for the next period**, choosing among a few redesign patterns.
Rewards are what that period's choice changed in the process Loss, against the
same episode run with the standard process.

## 3. Setup

### Disruption (one type first)
- **D1 offer dispatch outage:** offers cannot be sent (`complete → await_customer`
  blocked). This is the scenario where the process responds most to the agent's
  choices (always substitute 1.00 → 0.24).
- **Hidden length with a real dilemma.** Mixture: 60% short (4–24 h), 40% long
  (72–168 h). Short outages do almost no harm (waiting is right); long ones do a lot
  (the workaround is right). The agent only sees how long the outage has lasted so far.
- D2 and D3 come later, only once D1 works.

### Decisions
- Every **7 working hours** (twice per working day, calendar 07–21, Mon–Sat) while
  the outage lasts: about 1–3 decisions for a short outage, 6–14 for a long one.
- After the restore there is nothing left to decide. The episode runs on to the
  50-day horizon so that late harm is counted.

### Actions: process variants (redesign patterns)
| # | Variant | What the process does with blocked offers this period | Pattern |
|---|---|---|---|
| 0 | Standard process | wait for the dispatch system | — (reference) |
| 1 | Triage | urgent cases (slack < 7 days) go through the online-only channel; the rest wait | triage (Reijers & Limam Mansar 2005) |
| 2 | Alternative channel for all | every blocked offer goes online-only | alternative channel / workaround |
| 3 | Reserve team | backup workers send blocked offers by hand while the reserve budget lasts; the rest wait | extra resources (Reijers & Limam Mansar 2005) |

- Switching applies at once: cases already waiting are re-routed under the new variant.
- **The online-only channel already exists in the log** (`O_Sent (online only)`, 4.9%
  of offers). Its side effect (a share of customers miss it, so it is re-sent later)
  stays inside the simulation, so it needs no price.
- **The reserve team is a budget, not a price.** It is made of the real backup workers
  already mined for escalation, with at most B worker-hours per disruption (one
  assumption, swept). Using it early leaves nothing if the outage turns out long.
  Variant 3 is masked when the budget is spent.

### Reward
- Per period: r_t = −(Loss accrued in the period − the twin's Loss in the same
  period) / Loss_base. The last step also gets the rest of the horizon.
- The twin is the same episode with the standard process. Thanks to the separate
  random streams, it is the same world, so normal lateness cancels exactly.
- **No κ and no w.** Costs show up as Loss (re-sends, slower responses) or as the
  budget constraint.
- The return, with γ = 1, is −(Loss − Loss_twin)/Loss_base, i.e. the episode
  objective minus 1.

### Observation (about 10 features)
- time since the outage began; working hour and weekday;
- blocked cases, of which slack < 2 days and 2–7 days;
- Q and 24 h rolling Q;
- reserve budget left; reserve workers on shift and free;
- current variant.

The outage's remaining length stays hidden.

### Algorithm
Plain MaskablePPO with γ = 1. Episodes have 1–14 steps. `CaseMaskablePPO`, per-case
shares, β and the window are dropped.

## 4. Baselines and the go/no-go check

Before any training, measure how much room there is. All are cheap simulator runs:

1. **Static variants:** always 0 / 1 / 2 / 3.
2. **Timer rules:** "standard process, switch to variant v after X hours", with X
   swept (4, 8, 16, 24, 48 h). The best tuned timer rule is the strong baseline.
3. **Clairvoyant bound:** the best static variant chosen knowing the outage length.
   This is the value of information about the length.

**Go/no-go:**
- If the clairvoyant bound is clearly better than the best timer rule, there is room
  for a policy that reads the state, and DRL is justified.
- If not, a timer rule is already near-optimal. We then report that (with the
  analysis) instead of training, or add the second disruption type to create the
  dilemma.

## 5. What the result looks like

The trained policy maps a small state to one of four process variants. A decision
tree distils it into a DMN table, for example:

| Outage lasted | Blocked urgent cases | Reserve left | Variant |
|---|---|---|---|
| < 8 h | any | any | standard |
| 8–24 h | > 20 | > 0 | reserve team |
| 8–24 h | ≤ 20 | any | triage |
| > 24 h | any | any | alternative channel |

In BPMN, this is an event subprocess on the offer dispatch: a timer or condition on
"dispatch unavailable" evaluates the DMN table and routes blocked offers. That is a
resilience redesign artifact, which the per-case agent did not give us.

## 6. What changes and what stays

**Stays:**
- the validated simulator and its separate random streams;
- Q(t) and the threshold-free Loss;
- the do-nothing twin (here: the standard-process twin);
- the benchmark harness, training and evaluation infrastructure, TensorBoard.

**Goes:**
- per-case decisions, κ, w, β, the window, `CaseMaskablePPO`;
- per-scenario action semantics, D3/D4 in the first version.

**Remaining assumptions:**

| Assumption | Value | Note |
|---|---|---|
| Outage length distribution | 60% 4–24 h, 40% 72–168 h | swept |
| Online-only side effect | 20% re-send after 24–72 h | existing, assumed |
| Reserve budget B | e.g. 40 worker-hours per disruption | swept |
| Reserve handling time | 0.5–2 working h per offer | existing, assumed |
| Triage threshold | slack < 7 days | could be split into two triage variants |
| Decision period | 7 working hours | |

That is 6 assumptions, against about 25 in the per-case setup.

## 7. Work and run time

| Step | Effort |
|---|---|
| Simulator: process variant switch, re-routing waiting cases, reserve budget | ~½ day |
| Environment: period steps, period reward against the twin | ~½ day |
| Baselines: static, timer sweep, clairvoyant; go/no-go | ~2–3 h + runs |
| Training run | episodes take ~12 s plus the twin ~12 s. With 1–14 steps per episode, 20–30k steps take about 3–4 h on 6 cores. Caching the twin per training seed would roughly halve that |

## 8. Risks

- **A timer rule may already be near-optimal.** The go/no-go check finds this before
  any training is spent.
- **Few steps per episode means slow training per step.** It is a small problem
  (4 actions, ~10 features), so it should need far fewer samples than the per-case
  agent.
- **If the alternative channel has too little downside, "switch at once" dominates.**
  The mixture of short and long outages, the re-send effect and the budget are meant
  to prevent that. The go/no-go check shows whether they do.
