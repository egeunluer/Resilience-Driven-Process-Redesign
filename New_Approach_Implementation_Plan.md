# New Approach — Implementation Plan and Progress (branch `new-approach`)

White-box resilience engine on the original BPIC2017 log. Design rationale:
`drl_resilience_design_decisions.md`. Parameter sources: `Modules 2-3/Assumptions_Table.md`.

**Status legend:** ✅ done · 🔄 in progress · ⬜ not started

| Phase | Content | Status |
|---|---|---|
| 0 | Housekeeping: pending changes committed, assumptions table | ✅ |
| 1 | Small triage proof of concept | ✅ passed on 3 of 3 training seeds |
| 2 | Mine the full-log stage model | ✅ BPIC2017 mined; BPI2012 mined with the same code |
| 3 | Base simulator + validation against the log | ✅ BPIC2017 9/9 checks; BPI2012 runs unchanged |
| 4 | Q(t), Loss and recovery tracking | ⬜ |
| 5 | Disruptions D1, D2, D3, D6 (then D4, D5) | ⬜ |
| 6 | Recovery actions, scarce resources, costs | ⬜ |
| 7 | Environment | ⬜ |
| 8 | Calibration and baselines | ⬜ |
| 9 | Training | ⬜ |
| 10 | Distillation to BPMN + DMN | ⬜ |
| 11 | Experiments | ⬜ |

---

## Context

The thesis pivots away from resource allocation. A learned allocator converges to
"pick the fastest resource": on Consulta our best agent reached CPL ratio 0.54 against
0.47 for the fastest-resource rule. That's cycle-time efficiency, not resilience.

The new agent decides **which recovery action** a case gets when a disruption hits it,
sharing a scarce recovery capacity across many cases at once. The learned policy is
distilled into a BPMN + DMN decision table.

**Decisions taken**
- Data: the **original, full BPIC2017 log** (A_/O_/W_ events), not the W_-only subset.
- Q(t) = **share of open cases that are on track**.
- First action menu: **do-nothing, retry, reassign, substitute, escalate**.
- Disruptions D1–D6 below. Start with D1, D2, D3 and D6, then add D4 and D5.

**Log facts that drive the design** (from `Event Logs/BPI Challenge 2017.xes`)
- **Size:** 31,509 cases, 86 arrivals/day, about 1,900 open cases.
- **Cycle time:** median 19.1 days, p90 35.1 days, mostly customer waiting. Offer sent
  to returned: median 7.8 days. W_ work takes minutes per item.
- **Real 30-day offer expiry:** 76% of cancellations are automatic (`User_1`), 30.7 days
  after the last offer was sent.
- **Real missing-documents loop:** `A_Incomplete` in 48% of cases.
- **Real fraud specialist path:** `W_Assess potential fraud`, 1% of cases, 38 workers.
- **Real alternative offer channel:** `O_Sent (online only)`.
- **Automated steps:** 65% of applications are online and created by `User_1`.
- **Workforce:** about 38 workers active per weekday.

## Disruptions and the activities they hit

| ID | Disruption | Type | Real activities | What's mined vs assumed | Actions |
|---|---|---|---|---|---|
| D1 | Offer dispatch system outage | Service outage | `O_Create Offer`, `O_Created`, `O_Sent (mail and online)` | Mined: volumes, channel shares. Assumed: outage rate and repair time | do-nothing, retry, substitute (= `O_Sent (online only)`), escalate |
| D2 | Telephony / contact-centre outage | Service outage | `W_Call after offers`, `W_Call incomplete files` | Mined: call volumes, durations. Assumed: outage parameters | do-nothing, retry, substitute (written contact), escalate |
| D3 | Missing-documents surge | Data exception | `W_Validate application → A_Incomplete → W_Call incomplete files` | Mined: base rate per round, 1-day turnaround. Assumed: surge size | do-nothing, retry, reassign, escalate |
| D4 | Staff absence (later) | Resource unavailability | Validation and call teams | Mined: who can do what, calendars. Assumed: share and length | reassign, escalate |
| D5 | Fraud alert surge (later) | Exception | `W_Assess potential fraud` | Mined: base rate, team size. Assumed: surge size | do-nothing, escalate, reassign |
| D6 | SLA-breach risk | Deadline | Any open case near its deadline / the 30-day expiry | Mined: expiry, stage deadlines. Assumed: alert threshold | do-nothing, reassign, escalate |

## Q(t) — share of open cases on track

- A case is **on track** if its age ≤ `D_k`, the p90 (from the log) of case age when
  leaving its current stage k. For cases waiting on the customer, the real 30-day offer
  expiry counts.
- **Loss = ∫ max(0, Q_base − Q(t)) / Q_base dt**, integrated exactly. Normalised by the
  do-nothing Loss per scenario.
- **Hard penalties:** expiry auto-cancel, or a case finishing past its deadline.
- **Reported on top:** on-time completion rate, time to recover, number of auto-cancellations.
- **Why this definition:** cases take 20–35 days, so a completion-based measure would
  react weeks late. The share on track moves at once and uses all ~1,900 open cases.

## Architecture

A **new SimPy stage model** in `Modules 2-3/RIMS_DRL-main/loan_model/`. The RIMS Petri-net
`Token` isn't extended, for two reasons:
- Its control flow has only XOR weights: no timers, no races, no resource-free customer waits.
- Every RIMS agent step is a resource assignment.

RIMS files stay untouched, so the Consulta pipeline keeps working.

**Config-driven stage model (portability to other logs).** The simulator reads the process
from a stage-model config file instead of hard-coding loan origination. The code is
written once, and each event log is a config plus a mapping:
- **Log-independent (code):** simulator engine, disruption layer, Q(t) and Loss, recovery
  actions and scarce resources, environment, training, baselines, distillation, Module 1
  duration mining.
- **Log-specific (config, `Module 1/artifacts/<log>_stage_model.json`):**
  - stages;
  - transitions with mined probabilities;
  - resource-free waits (empirical quantiles);
  - work items (which W_-type activities need a worker);
  - timers and races (e.g. the 30-day offer expiry);
  - loops;
  - stage deadlines `D_k`;
  - which disruptions can hit which stage.
- **Mapping file (`<log>_activity_map.json`):** log activity names → stages and roles.
  - **BPIC2017 → BPI2012:** the same bank and process, so only a mapping is needed
    (`W_Nabellen offertes` ↔ `W_Call after offers`, `O_SENT_BACK` ↔ `O_Returned`, …)
    plus a rerun of the mining.
  - **A different process** (Helpdesk, Consulta, …): a new config, no new code. Timers and
    races have to be identified by hand, since they can't be mined automatically.
- **Thesis value:** this also supports doc §14's requirement of several BPIC logs to show
  the method generalises.

**Reused**
- **Worker calendars:** `RoleSimulator` (`role_simulator.py`).
- **Arrival profile:** empirical weekly arrivals (`mine_arrival_profile.py`, `inter_trigger_timer.py`).
- **W_ durations and qualified workers:** Module 1 per-(resource, W_ activity) durations
  (`bpi2017.json`, `bpi2017_min3activities_pool.json`).
- **Shock schedule:** `shock.py` (cycle schedule, `shock_key`).
- **Area integration:** the exact integration from `gym_env._integrate_phi_area`.
- **Training infrastructure:** `callbacks.py` and the `train_DRL.py` setup (MaskablePPO,
  constant LR, resume, TensorBoard).
- **Benchmark harness:** the paired benchmark pattern.

## Phases

### Phase 0 — Housekeeping ✅
- Constant-LR and 32-episode evaluation changes committed on `main` (84ae2fe, not pushed);
  `new-approach` fast-forwarded to it.
- `Modules 2-3/Assumptions_Table.md` created; filled in as phases progress.

### Phase 1 — Small triage proof of concept ✅
- **Goal:** check cheaply that DRL is justified before the full build (doc §16).
- **Code:** `loan_model/toy.py` and `toy_poc.py`.
- **World:**
  - 30 cases with their own deadlines;
  - 1 outage of uncertain length (short 1–4 h or long 12–36 h);
  - 1 escalation slot and 1 backup server shared by all cases;
  - substitute can cause rework later.
- **Pass criterion:** PPO beats always-retry, always-escalate and greedy on paired seeds,
  with a 95% interval above zero.
- **Seeds:** calibration 0–15, checkpoint selection 1000–1063, benchmark 200–455.
- **Result (training seed 0, 3M steps, selected checkpoint 1.75M):** PASSED.
  - vs greedy: +0.16 ± 0.16 (borderline);
  - vs always-escalate: +7.5;
  - vs always-retry: +13.2;
  - vs do-nothing: +2.9.
  
  PPO uses the escalation slot more selectively than greedy. It misses fewer deadlines
  (2.30 vs 2.93) and has lower Loss (0.27 vs 0.46), at a slightly higher action cost.
- **Lessons for the full model:**
  1. An entropy bonus and reward normalisation are needed; without them PPO collapsed to do-nothing.
  2. The observation must contain system urgency (other open cases near their deadline,
     arrivals still to come); without it PPO stayed below greedy.
  3. The exact Loss integral matches a brute-force integral.
- **Robustness: 3 training seeds** (3M steps each, checkpoint selected on seeds 1000–1063,
  benchmark on 256 paired episodes); all three pass:

  | Training seed | Return | Loss ratio | Missed | vs greedy | vs always-escalate | vs always-retry |
  |---|---|---|---|---|---|---|
  | 0 | −5.21 | 0.272 | 2.30 | +0.16 ± 0.16 | +7.5 ± 1.0 | +13.2 ± 2.0 |
  | 1 | −5.09 | 0.246 | 2.16 | +0.28 ± 0.16 | +7.6 ± 1.0 | +13.3 ± 2.0 |
  | 2 | −5.20 | 0.254 | 2.26 | +0.17 ± 0.16 | +7.5 ± 1.0 | +13.2 ± 2.0 |
  | greedy (for reference) | −5.37 | 0.456 | 2.93 | — | — | — |

### Phase 2 — Mine the full-log stage model ✅
- **Define the stage-model config schema** (see "Config-driven stage model" above) and
  write `bpic2017_activity_map.json` (activity → stage / role).
- **Mining script:** generic, driven by the mapping (`loan_model/mine_stage_model.py`), so
  the same script later mines BPI2012 from `bpi2012_activity_map.json`.

The mining script writes `Module 1/artifacts/bpic2017_stage_model.json` (planned contents):
- **Arrivals:** hour-of-week profile; online share.
- **Routing:**
  - online → `W_Handle leads`;
  - offers per case;
  - `O_Sent` channel split;
  - customer returns vs cancels;
  - `A_Incomplete` per validation round;
  - fraud flag;
  - final outcome.
- **Resource-free delays** as empirical quantiles: customer response, documents back,
  customer cancellation timing.
- **Expiry:** 30-day offer expiry.
- **Work:** W_ work durations and qualified workers (Module 1). A_/O_ milestones take zero time.
- **Q(t) deadlines:** stage deadlines `D_k`.

**Done (outputs):**
- `loan_model/STAGE_MODEL_SCHEMA.md`: config schema and how the simulator reads it.
- `Module 1/artifacts/bpic2017_activity_map.json` (by hand) and
  `loan_model/mine_stage_model.py` (generic miner) → `bpic2017_stage_model.json`.
- **Model:** 5 stages
  `intake → complete → await_customer → validation ⇄ incomplete → accepted / denied / cancelled`.
  Each stage has one work item, e.g. `W_Validate application`. Mined:
  - routing per visit number;
  - work sessions per worker;
  - resource-free waits (initial delay, gaps, tail);
  - the offer-expiry timer (30.7 d, explains 85% of await_customer cancellations);
  - stage deadlines;
  - calendar (Mon–Sat 07–21 local);
  - weekly arrival profile;
  - validation targets.
- **Work is mined as active sessions** (start/resume → suspend/complete/abort), not with
  Module 1's start→complete pairing: BPIC2017 work items are suspended for days while
  waiting on the customer, so that pairing mixes waiting into work time. Module 1's W_
  durations are therefore not reused.
- **Portability check passed early:** `bpi2012_activity_map.json` runs through the same
  miner (BPI2012: 13,087 cases, 53 workers). It needed two small, generic additions:
  - entry keys of the form `ACTIVITY@lifecycle` (BPI2012 marks validation only by
    `W_Valideren aanvraag@schedule`);
  - skipping sessions without a recorded resource.
  
  BPI2012 also shows the **same 30.7-day offer expiry**, an independent confirmation of
  the timer.
- **Open points for Phase 3/4:**
  1. The mined waits include the real queueing time, so the simulator's own queueing may
     double-count it. Phase 3 validation checks this.
  2. Intake's p90 deadline is 7.8 min, too tight to be meaningful; set a floor in Phase 4.

### Phase 3 — Base simulator + validation ✅
- **`loan_model/simulator.py`:** a **generic stage-model engine** that reads the config,
  with a SimPy process per case. Nothing is specific to BPIC2017 in the code.
  - Worker pools with calendars.
  - A fixed dispatcher (per-activity FIFO, fastest free qualified worker).
  - Customer delays and the expiry race.
  - An event log for comparison with the real one.
  - About 40 simulated days of warm-up until ~1,900 cases are open. The simulation starts
    empty, while the real bank always has ~1,900 open cases spread over all stages; cases
    live 19–35 days, so the warm-up must run about one case lifetime.
  - Warm-up cost is measured here. **Fallback if too slow:** start from the real log's open
    cases on a given date (stage and age from the log; "short-term simulation", Rozinat et
    al. 2009), or simulate the warm-up once per seed and reuse the state.
- **`validate_loan_model.py`** checks against the log:
  - cycle-time distribution;
  - outcome shares;
  - incomplete share and validation rounds;
  - offers per case;
  - open cases;
  - worker utilisation;
  - weekly pattern.
- **Also measure here:** how heavily workers are loaded, which decides whether D4 can bite.
- **Must pass before any disruption is added.**

**Done (outputs):**
- `loan_model/simulator.py`: a generic engine that reads the stage-model JSON (no
  BPIC2017-specific code).
  - Workers are capacity-1 servers, present per day with their mined probability, on
    shift Mon–Sat 07–21 local.
  - Fixed dispatcher: fastest free qualified worker, FIFO queue per work item.
  - The offer-expiry timer races the customer wait (`simpy.AnyOf`).
  - Records an event log (stage entries, sessions with request time).
- `loan_model/calendar_time.py`: working-time clock shared by miner and simulator.
- `validate_loan_model.py`: simulated vs log comparison (3 seeds × 240 days, cases
  arriving on days 45–150).
- **Fix found by validation:** with wall-clock waits the cycle-time median was +14%,
  because waits ending at night queued until the morning shift, counting off-hours twice
  (the log's waits always end inside a shift). Waits that end in a session are now mined
  and replayed in **working time** (`*_w` fields). After the fix, queueing for workers
  is ~0.

**Validation result, BPIC2017 (all 9 checks ok):**

| Metric | Simulated | Log |
|---|---|---|
| Cycle time p50 / p90 | 20.4 / 36.4 d | 19.1 / 35.0 d |
| KS distance (cycle time) | 0.066 | — |
| Accepted / cancelled / denied | 53.6 / 34.3 / 12.0 % | 54.7 / 33.1 / 11.9 % |
| Cases with incomplete loop | 46.3 % | 47.6 % |
| Validation rounds 0 / 1 / 2 / 3 | 31.8 / 32.1 / 24.5 / 8.4 % | 30.6 / 32.4 / 25.0 / 8.6 % |
| Offers per case 1 / 2 / 3 | 72.5 / 21.2 / 4.3 % | 72.8 / 20.9 / 4.3 % |
| Open cases (median) | 1,904 | 1,937 |
| Arrivals per day | 86.0 | 86.2 |

**Findings that matter for the next phases:**
- **Warm-up:** open cases reach 95% of steady state on day 35, so 40 days is enough.
  A 40-day warm-up takes ~1.1 s, so no shortcut is needed for now (revisit if training
  needs many thousands of episodes).
- **Worker utilisation is ~4%.** The log only records the minutes a work item is open
  (~13 min of logged work per case, 18.6 h per day for ~40 workers). Capacity is
  therefore not a bottleneck. **D4 (staff absence) would only bite if almost the whole
  team were absent**, so it stays deprioritised. D1/D2/D3/D6 don't depend on worker
  capacity. The scarce resource for triage is the escalation team (Phase 6).
- **Open-case level:** the median matches, but the log's spread (p10–p90 1,507–2,343)
  comes from month-to-month swings in arrivals. The weekly arrival profile is
  stationary, so the simulated spread is narrower (1,851–1,964). Acceptable for one-shock
  episodes; seasonal level could become a scenario parameter later.
- **Stage-level gaps** (cycle time is fine overall): validation visit median 0.50 vs
  0.86 d, incomplete median 1.83 vs 1.09 d. These come from sampling the waits within a
  visit independently; to look at if a disruption targets these stages specifically.
- **BPI2012 portability:** mined, simulated and validated with the same code. KS 0.035;
  outcomes within 2 pp; open cases 731 vs 726; cycle-time p50 / p90 0.91 / 27.0 vs
  0.81 / 30.5 d.

### Phase 4 — Q(t), Loss and recovery tracking ⬜
- **`loan_model/metrics.py`:** on-track status, Q(t), exact Loss, penalties.
- **Recovery:**
  - the disruption has ended;
  - all cases it hit have left the affected stage;
  - Q(t), averaged over one working day, is back at or above Q_base − tolerance.
- **Episode horizon:** capped at 60 days after onset.

### Phase 5 — Disruptions ⬜
- **`loan_model/disruptions.py`:** D1, D2, D3, D6 first; one scenario per episode,
  cycling through types and strengths.
- **Failure hook:** a failed step creates a decision event, and the case waits until the
  agent decides.

### Phase 6 — Recovery actions, scarce resources, costs ⬜
- **`loan_model/recovery.py`:**
  - **do-nothing / retry / substitute** as in the proof of concept.
  - **reassign:** the case moves to the front of its queue with the fastest free qualified worker.
  - **escalate:** uses the scarce senior pool (capacity 1–3); masked when no slot is free.
- **Costs:** κ = retry 0.05, reassign 0.15, substitute 0.30, escalate 0.50 (× c_p, swept).

### Phase 7 — Environment ⬜
- **`loan_env.py`:** gymnasium Env, MaskablePPO compatible. `reset` runs the warm-up, then
  the scenario, then runs until the first decision.
- **Observation:**
  - outage flags and elapsed time;
  - free workers, escalation slots, open cases;
  - Q(t) / Q_base;
  - **system urgency aggregates** (Phase 1 lesson);
  - for the deciding case: stage, failure type, slack, days left until expiry, attempts so far.
- **Reward:** `−(w · ΔLoss / Loss_base + Σ κ · c_p + penalties)`.
- **Episode-end info:** `info['resilience']`, for the existing callbacks.

### Phase 8 — Calibration and baselines ⬜
- **`calibrate_loan_resilience.py`:** Q_base and the do-nothing `Loss_base` per scenario.
- **`baselines_loan.py`:**
  - do-nothing;
  - always-X;
  - greedy;
  - hand-written rules (SmartPM / P20 style);
  - a Metzger-style binary trigger.
- **Benchmark:** paired, 64+ episodes.

### Phase 9 — Training ⬜
- **`train_loan_DRL.py`:** MaskablePPO with constant LR, entropy bonus and reward normalisation.
- **Evaluation and selection:** 32-episode fixed-seed evaluation; keep every checkpoint
  and pick the final model on a separate seed set.

### Phase 10 — Distillation ⬜
- **`distill_policy.py`:**
  - one decision tree per failure type (depth 3–5);
  - fidelity, and performance when the tree itself is the policy;
  - DMN 1.3 XML export plus a BPMN fragment.

### Phase 11 — Experiments ⬜
- **Strength sweeps:** curves of Loss vs disruption strength.
- **Out-of-distribution test:** held-out combined scenarios.
- **Cost sweeps:** κ and w sweeps, giving the resilience-vs-cost Pareto.
- **Seeds:** 3 or more training seeds, significance tests.

## Verification
- **Phase 1:** paired benchmark vs baselines, over 3 training seeds.
- **Phase 2/3 portability check:** the BPI2012 mapping + mining runs through the same scripts
  and simulator without code changes. Validating BPI2012 fully against its log comes later.
- **Phase 3:** simulator matches the log (cycle-time median / p90 within ~10%, outcome
  shares within ~3 percentage points, open cases ≈ 1,900); reviewed with the user.
- **Phases 4–7:**
  - smoke tests: do-nothing episodes recover and Loss > 0;
  - masks never empty;
  - Loss equals a brute-force integral;
  - fixed seeds give identical episodes.
- **Phases 8–9:** baseline ladder, then the agent on the paired benchmark.
- **Phase 10:** fidelity ≥ 90%, tree-policy benchmark, the DMN loads in a viewer.
