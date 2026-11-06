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
| 4 | Q(t), Loss and recovery tracking | ✅ exact tracker, calibrated Q_ref, recovery tested |
| 5 | Disruptions D1, D2, D3, D6 (then D4, D5) | ✅ fixed 50-day episodes; customer patience deferred |
| 6 | Recovery actions, scarce resources, costs | ✅ implemented and checked with fixed policies |
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

### Phase 4 — Q(t), Loss and recovery tracking ✅
- **`loan_model/metrics.py`:** on-track status, Q(t), exact Loss, penalties.
- **Recovery:**
  - the disruption has ended;
  - all cases it hit have left the affected stage;
  - Q(t), averaged over one working day, is back at or above Q_base − tolerance.
- **Episode horizon:** capped at 60 days after onset.

**Done (outputs):**
- `loan_model/metrics.py` (`QTracker`): an observer of the simulator. It tracks Q(t)
  exactly; deadline crossings sit in a heap and are processed lazily.
  - Q, Loss and the 24 h rolling Q match brute-force integrals on fine grids (to the
    second, and to 5 decimals).
  - It counts expiries caused by delay (the timer won the race).
  - `hit()` / `clear_hit()` mark blocked steps.
- `calibrate_loan_resilience.py` → `Module 1/artifacts/bpic2017_resilience_calibration.json`
  (16 seeds × 120 days, 1.5 min).

**Decisions (made with data):**
1. **Q(t) as agreed:** case age ≤ D_k of its current stage, over all open cases.
   Compared with three alternatives on a 3-day full work stop (day 70, steady state):

   | Variant | Daily noise (sd) | Drop, 1-day stop | Drop, 3-day stop | S/N (3-day) |
   |---|---|---|---|---|
   | age, all open (chosen) | 0.0073 | 0.033 | 0.066 | 9.1 |
   | time in stage, all open | 0.0094 | 0.035 | 0.079 | 8.5 |
   | age, bank stages only | 0.0117 | 0.060 | 0.072 | 6.1 |
   | time in stage, bank stages only | 0.0182 | 0.061 | 0.097 | 5.3 |

   Restricting Q to the bank's own stages makes it drop more but also adds as much noise.
   About 1,300 of the ~1,900 open cases wait on the customer, so they dilute Q; this is
   a property of the process.
2. **Deadline floor of 24 h:** only affects intake (p90 age at exit = 7.8 min).
3. **Loss below a normal-operation floor `Q_ref`**, not below the mean:
   - Without disruption, Q has mean 0.904 and daily sd 0.009.
   - **Q_ref = 0.890**, the 5th percentile of daily means.
   - **Loss = ∫ max(0, Q_ref − Q) / Q_ref dt.**
   - Baseline Loss without disruption is ~12 s per day, i.e. noise barely counts.
4. **Hard penalty only for expiries caused by delay**, the 30-day timer firing because a
   case was held up. Baseline: 1.4 per day, from customer responses sampled just past the
   timer. Customers who never respond are not penalised.
5. **Recovery:**
   - the disruption is over;
   - every blocked step has been carried out (cleared by the disruption layer; leaving the
     stage clears it as a fallback);
   - the 24 h rolling Q is ≥ Q_ref.
   
   Clearing hits only when the case leaves its stage was too strict: cases in validation /
   incomplete wait on the customer for weeks.

**End-to-end check** (Monday work stop, 4 seeds):

| Stop | Recovery after | Loss at recovery | Delay expiries |
|---|---|---|---|
| 0 d | immediately | 0 | 0–1 |
| 1 d | 1.6 d | 0.003 Q-days | 1–5 |
| 3 d | 4.2 d | 0.046 Q-days | 3–7 |

**Open point:** some effects come after recovery (delayed cases crossing deadlines later,
expiries ~30 days on). The expiry count can be followed past the episode end in the
evaluation (Phase 8) if it matters.

### Phase 5 — Disruptions ✅
- **`loan_model/disruptions.py`:** D1, D2, D3, D6 first; one scenario per episode,
  cycling through types and strengths.
- **Failure hook:** a failed step creates a decision event, and the case waits until the
  agent decides.

**Done so far:**
- **Constrained pool option** (`StageModel(..., pool=top_k_pool(model, k))`). It still
  passes validation down to the top-3 workers per work item:

  | Pool | Workers | Utilisation | Cycle time p50 / p90 | Checks passed |
  |---|---|---|---|---|
  | full | 145 | 4% | 20.4 / 36.4 d | 9/9 |
  | top 10 per work item | 39 | 11% | 20.1 / 36.4 d | 9/9 |
  | top 5 per work item | 24 | 19% | 20.8 / 36.6 d | 9/9 |
  | **top 3 per work item** (working default) | **16** | **36%** | 20.8 / 36.4 d | 9/9 |
  | top 2 per work item | 10 | 63% | 21.2 / 36.9 d | 8/9 |

  With the top-3 pool, queues start to form (8–9% of requests wait over an hour).
  Q_ref is still from the full pool and must be recalibrated for the chosen pool.
- `Module 1/artifacts/bpic2017_disruptions.json`: log-specific scenarios, 2 levels each.
  - **D1 offer dispatch outage:** blocks the complete → await_customer transition.
  - **D2 telephony outage:** blocks `W_Call after offers` and `W_Call incomplete files`.
  - **D3 missing-documents surge:** validation → incomplete × (1 + m).
  - **D6 SLA-risk alert:** a blocked, waiting case gets a second decision at 48 h slack.
  - Outage lengths come from a short/long mixture the agent never sees.
- `loan_model/disruptions.py`: generic mechanisms (outage, routing surge, alert).
- **Simulator hooks:** blocked steps create `Decision`s, answered by a policy or
  queued in `sim.pending` for the environment.
- `loan_model/episode.py`: warm-up, onset at a random time in the following week, run
  until recovery (60-day cap).

**Smoke test (do-nothing, top-3 pool, 6 seeds):**
- Every scenario creates 200–700 decisions on 130–690 cases per episode, which meets
  the "many concurrent cases" requirement.
- All episodes recover; an episode takes ~1.3 s.

**Problem found — impact:**
- Up to the declared recovery, Loss ≈ 0 for everything except D1 level 2. The delays are
  1–3 days, the age deadlines are 28–35 days, so delayed cases only cross their
  deadlines weeks after the episode has ended.
- Measured over a fixed 45 days after onset (8 paired seeds, vs no disruption):

  | Scenario | Extra Loss | Extra delay expiries | Extra late completions |
  |---|---|---|---|
  | D1:2 | +0.186 ± 0.143 Q-days | +1.0 | +17 |
  | D2:2 | +0.098 ± 0.060 Q-days | +7.8 | +58 |
  | D3:2 | +0.012 ± 0.033 (n.s.) | +2.9 | +18 |

  The harm exists for D1 and D2 but arrives late. D3 is close to harmless in this model.
**How long the effects last** (strongest levels, 12 paired seeds, 70 days, do-nothing):

| Scenario | When the harm happens | 95% of total harm reached by |
|---|---|---|
| D1:2 | week 1 | day 6 |
| D2:2 | weeks 5–6, when delayed cases reach their deadlines | day 40 |
| D3:2 | weeks 3–6 | day 33 |

From week 7 on, the differences are only noise.

**Decisions:**
1. **Episodes end at a fixed horizon of 50 days after onset**, not at recovery. 50 rather
   than 45 days gives margin for the longest outages, which shift D2's tail by a few days.
   Time to recover is still reported. Decisions happen only during the disruption (and
   alerts); the episode then runs on to the horizon so late harm is counted.
2. **D3 stays as it is for now.** Its effect is small (n.s. at 45 days).
3. **Q_ref recalibrated for the top-3 pool:** 0.886 (mean 0.901, daily sd 0.010;
   `bpic2017_pool3_resilience_calibration.json`).

**Deferred option (documented, not implemented): customer patience.** Today, whether a
customer returns the offer or cancels is drawn on stage entry, regardless of delay, so a
disruption delays cases but never loses a customer. The option:
- each case accumulates "disruption delay": time blocked, waiting for an outage to end,
  retrying, or queuing for escalation;
- when the block ends, the customer cancels with probability 1 − exp(−β · delay); the case
  ends as `END_cancelled` and counts as a lost customer (hard penalty);
- β is assumed and swept (e.g. 0 / 5 / 15 % per day of delay); β = 0 is the current model;
- rough anchor: about 24% of BPIC2017 cancellations are made by a person rather than the
  timer, and their timing relative to waiting time can be mined (observational, so β
  stays a swept assumption).

Effect if switched on: harm appears during the disruption instead of weeks later, D2/D3
matter more, and the agent's actions change how many customers are lost. To reconsider
if the agent's learning signal turns out too weak with the 50-day horizon alone.

### Phase 6 — Recovery actions, scarce resources, costs ✅
- **`loan_model/recovery.py`:**
  - **do-nothing / retry / substitute** as in the proof of concept.
  - **reassign:** the case moves to the front of its queue with the fastest free qualified worker.
  - **escalate:** uses the scarce senior pool (capacity 1–3); masked when no slot is free.
- **Costs:** κ = retry 0.05, reassign 0.15, substitute 0.30, escalate 0.50 (× c_p, swept).

**Done (outputs):**
- `"recovery"` block in `bpic2017_disruptions.json` (all values assumed, swept):
  - **costs** κ: retry 0.05, reassign 0.15, substitute 0.30, escalate 0.50;
  - **allowed actions** per failure type; an SLA-risk alert inherits the actions of the
    failure that blocked the case;
  - **retry:** 1 working hour; a re-fetch finds missing documents with p = 0.4;
  - **substitute:**
    - D1: online-only channel, 20% chance the offer has to be re-sent 1–3 days later;
    - D2: written contact, customer responds 1–3 days later;
  - **escalate:** senior team, 2 shared slots, 0.5–2 working hours per case, success 0.95,
    masked when no slot is free;
  - **reassign:** the case's incomplete-file calls go to the front of the queue.
- **Simulator:**
  - `attach_disruption()`, `allowed_actions()` (the action mask);
  - the action effects (`_recover`, `_documents_decision`, `_escalate`);
  - the escalation team as a shared `simpy.Resource`;
  - action-cost accounting.
- **Bug fixed:** results changed between processes. The dispatcher looped over each
  worker's work items as a *set*, whose order depends on the per-process hash seed, and
  ties between equally old requests are common (many waits end exactly at 07:00). Now
  sorted. Results are identical across hash seeds, and identical policies give
  identical results.

**Fixed-policy check (level 2, top-3 pool, 8 paired seeds, 50-day horizon):**

| Scenario | do-nothing Loss | Best action | Best Loss (vs do-nothing) |
|---|---|---|---|
| D1 offer outage | 0.124 | substitute | 0.006 (−0.118 ± 0.117) |
| D2 telephony outage | 0.053 | substitute | 0.017 (−0.037 ± 0.050, n.s.) |
| D3 documents surge | 0.006 | none helps | no difference |

Other findings:
- **Escalation** helps a little but is capped by its 2 slots (~65 escalations per episode).
- **Retrying every hour during a long outage** brings nothing and creates up to 24,000
  decisions per episode.

**Follow-up (decided with the user):**
- **Retries capped at 2 per case** (`retry.max_per_case`; retry is masked afterwards).
  Decisions per episode drop from up to 24,000 to ~1,100 (D1) and ~2,100 (D2).
- **D3 made stronger.** The base incomplete share is already 0.69 on the first round, so
  raising it changed little. The faulty portal now also holds up the customer's upload
  (level 1: 24–96 h, level 2: 72–168 h). Reassign means a caseworker chases the
  documents directly, so the case skips that delay.
- **Rerun (level 2, 8 paired seeds):**

  | Scenario | do-nothing Loss | Best fixed action | Its Loss (vs do-nothing) |
  |---|---|---|---|
  | D1 offer outage | 0.124 | substitute | 0.006 (−0.118 ± 0.117) |
  | D2 telephony outage | 0.053 | substitute | 0.017 (−0.037 ± 0.050, n.s.) |
  | D3 documents surge | 0.048 | retry (re-fetch) | 0.010 (−0.039 ± 0.032) |

  Reassign also helps on D3 (0.016), and retry no longer hurts D1/D2 much. Each scenario
  now has a different best action, which is what the agent has to learn.
- **Reward scale:** settled in Phase 7, below.
- **Variance from outage length:** settled in Phase 8 (fixed length cycle), below.

### Phase 7 — Environment ✅
- **`loan_env.py`:** gymnasium Env, MaskablePPO compatible. `reset` runs the warm-up, then
  the scenario, then runs until the first decision.
- **Observation:**
  - outage flags and elapsed time;
  - free workers, escalation slots, open cases;
  - Q(t) / Q_base;
  - **system urgency aggregates** (Phase 1 lesson);
  - for the deciding case: stage, failure type, slack, days left until expiry, attempts so far.
- **Reward** for the step between decision t and the next one:
  `r_t = −( w · ΔLoss_t / Loss_base[e, ℓ]  +  κ_a · c_p[e] )`, with γ = 1.
  - **Loss term:** ΔLoss_t is the Loss accrued since the last decision. It is divided by
    the do-nothing Loss of the same scenario e and outage length ℓ (Phase 8), so a
    do-nothing episode sums to −w and a perfect one to 0, whatever the scenario. The
    Loss after the last decision, up to day 50, goes into the last step's reward.
  - **Cost term:** `c_p[e] = 1 / N_e`, where N_e is the mean number of decisions per
    episode under do-nothing. Acting on every case then costs κ in total: substitute
    everywhere ≈ 0.3, escalate everywhere ≈ 0.5, retry ≈ 0.05. Costs are then on the
    same 0–1 scale as the normalised Loss. With the Phase 6 numbers (w = 1):
    - D1: substitute everywhere ≈ 0.05 Loss + 0.17 cost = 0.22, against 1.0 for doing nothing;
    - D2: substitute ≈ 0.31 + 0.35 = 0.66, a real trade-off;
    - D3: retry ≈ 0.20 + 0.05 = 0.25.
  - **w** sets resilience against cost. Default 1, swept in Phase 11 for the Pareto front.
  - **No separate expiry penalty.** Expiries are ~70 per episode under every policy:
    almost all are the normal 30-day expiry, not caused by the disruption. Their harm
    is already in Loss.
  - Reward normalisation (VecNormalize) is kept on top, as in the proof of concept.
- **Episode-end info:** `info['resilience']`, for the existing callbacks.


**Done (outputs):**
- **`loan_model/episode.py`:** an `Episode` class, shared by `run_episode` (fixed policy)
  and the environment:
  - warm-up, onset and fixed horizon;
  - hourly recovery check and the lowest 24 h rolling Q;
  - `run_to_decision()` stops the simulation exactly when a decision is pending.
  - **Onset now drawn in working time:** an outage that falls entirely at night blocks
    nothing and would give an episode with no decision. Episodes that still have no
    decision are redrawn and counted (`empty_episodes`).
- **`loan_model/disruptions.py`:** an optional fixed `duration_h` (for the Phase 8 cycle);
  cell keys such as `D1_offer_outage:2:24h`.
- **Simulator:**
  - a `decision_ready` event, so the environment can wait for the next decision;
  - the case's offer-expiry time (`expiry_at`), which the agent observes.
- **`loan_env.py`:** `LoanEnv` (gymnasium, MaskablePPO `action_masks()`):
  - **actions:** 5 (do-nothing, retry, reassign, substitute, escalate);
  - **observation:** 33 features in [−1, 1]:
    - disruption on, time since onset, time since its end;
    - working hours, hour of day;
    - free escalation slots, pending decisions, blocked cases;
    - share of blocked cases with slack < 2 d and 2–7 d;
    - Q / Q_ref and rolling Q / Q_ref;
    - open cases;
    - scenario, cause and alert flags;
    - for the case: stage, slack, age, time to offer expiry, attempts, retries left,
      validation round, offers, fraud flag;
  - **reward:** as above. Needs `Loss_base` and `N` per cell from Phase 8 (or `norm=`);
    the environment refuses to start without them;
  - **scenario cycle** `(scenario, level, duration_h)` with `shock_cycle_index`, as
    `EvalPolicyCallback` expects;
  - **episode end:** `info['resilience']` and `last_episode_info` with objective,
    loss_ratio, cost_ratio, Loss, time to recover, min rolling Q, expiries, decisions and
    counts per action. `mean_cycle_time` returns the objective, so `EvalPolicyCallback`
    can select on it (renamed properly in Phase 9).
  - **After an action, the case acts on it at once.** An escalation then holds its slot
    before the mask of another decision at the same instant is computed.

**Checks (placeholder normalisation from the Phase 6 level-2 do-nothing runs):**
- **Same results as the fixed-policy runner:** do-nothing through the environment gives
  exactly the Loss and decision count of `run_episode` (3 scenarios).
- **Rewards add up:** the per-step rewards sum to −objective.
- **Masks and observations:** masks never empty, observations always in bounds.
- **Reproducible:** same seed and actions give an identical episode, also across hash seeds.
- **Speed:** ~2.8 s per episode, ~250 steps/s with MaskablePPO, so ~35 min per 500k steps.
- **MaskablePPO + VecNormalize + `ResilienceCallback`** run without errors.
- **Gymnasium's `check_env` does not apply:** it resets twice expecting the same episode
  (the cycle moves on) and steps with masked actions (rejected on purpose).

### Phase 8 — Calibration and baselines ✅
- **Fixed outage-length cycle** (like the demand-spike magnitudes in the previous
  approach). In training, the disruption length is no longer drawn from the mixture.
  Episodes cycle through a fixed grid of (scenario, length) pairs, e.g.:
  - D1, D2: 8, 24, 72, 168 h;
  - D3 surge: 72, 168, 240 h.

  The agent still does not observe the length: it only sees the elapsed time, so
  "wait or substitute?" remains a decision under uncertainty. The cycle:
  - covers short and long outages equally in every batch;
  - lets `Loss_base` be calibrated per (scenario, length), so the reward is normalised
    against the matching do-nothing episode, which removes most of the length-driven
    variance;
  - makes evaluation and benchmark episodes comparable cell by cell, with paired seeds
    per cell.

  The random lengths of the mixture are kept as a robustness check on the benchmark
  (lengths between and beyond the grid).
- **`calibrate_loan_resilience.py`:** Q_base, `Loss_base[e, ℓ]` and `N_e` (decisions per
  episode) under do-nothing for each cell of the cycle (seeds 0–15).
- **`baselines_loan.py`:**
  - do-nothing;
  - always-X;
  - greedy;
  - hand-written rules (SmartPM / P20 style);
  - a Metzger-style binary trigger.
- **Benchmark:** paired, 64+ episodes.


**Done (outputs):**
- **Training cycle** in `bpic2017_disruptions.json` (`training_cycle`), 11 cells:
  - D1 and D2 at 8, 24, 72 and 168 h;
  - D3 surge at 72, 168 and 240 h.

  Plus `robustness_cells`: the same scenarios with the length drawn from level 2.
- **`calibrate_loan_resilience.py --scenarios`:** do-nothing Loss_base and N per cell
  (seeds 0–15), plus their mean per scenario over the cycle.
- **`baselines_loan.py`:** do-nothing, always-X, greedy (myopic and deadline-aware),
  rule (SmartPM / P20 style), and a paired benchmark on seeds 200+.
- **Metzger binary trigger:** needs training, so it moves to Phase 9.

**Calibration (do-nothing, top-3 pool, 16 seeds):**

| Cell | 8 h | 24 h | 72 h | 168 h |
|---|---|---|---|---|
| D1 offer outage, Loss (Q-days) | 0.005 | 0.005 | 0.008 | 0.497 |
| D2 telephony outage, Loss (Q-days) | 0.007 | 0.009 | 0.015 | 0.178 |
| D3 surge (72 / 168 / 240 h), Loss (Q-days) | — | — | 0.016 / 0.026 | 0.063 (240 h) |

- **Short outages are almost harmless.** Up to 72 h, the Loss stays at the level of normal
  day-to-day noise (~0.01 over 50 days, sd ~0.01): blocked cases are held for at most
  3 days, and the age deadlines (p90) leave that much slack. Only the week-long outages
  hurt.
- **So the reward is normalised per scenario, not per cell.** It uses the cycle mean:
  D1 0.129, D2 0.052, D3 0.035 Q-days; N = 451 / 590 / 474. Dividing by a short cell's
  own Loss_base would blow pure noise up ~100×. Short outages now give small rewards,
  correctly, because little is at stake.
- **Pairing and VecNormalize** handle the remaining variance.

**Baseline benchmark (11 cycle cells × 8 seeds, paired; objective = Loss/Loss_base +
cost/N, do-nothing ≈ 1, lower is better):**

| Policy | Objective | vs do-nothing (95% CI) | Loss part | Cost part |
|---|---|---|---|---|
| do-nothing | 1.017 | — | 1.017 | 0 |
| always retry | 0.914 | −0.10 ± 0.16 | 0.848 | 0.066 |
| always reassign | 1.013 | −0.00 ± 0.16 | 0.966 | 0.047 |
| always substitute | 0.608 | −0.41 ± 0.27 | 0.422 | 0.186 |
| always escalate | 0.774 | −0.24 ± 0.18 | 0.707 | 0.067 |
| greedy | 0.748 | −0.27 ± 0.25 | 0.717 | 0.032 |
| **rule** | **0.583** | **−0.43 ± 0.22** | 0.450 | 0.132 |

- **No single policy wins every cell:**
  - short outages: do-nothing (substituting wastes money);
  - D1 168 h: substitute (0.49);
  - D3: retry or escalate.

  Picking the best fixed policy per cell gives ~0.34. That bound knows the outage
  length, which the agent does not, but it shows the room left above the rule (0.58).
- **The rule is the baseline to beat.** It is also best on the robustness cells (0.26 vs
  do-nothing 0.34).
- **Noise:** with 88 paired episodes, the 95% CI is still ±0.2, driven by the
  week-long cells. The Phase 9 benchmark should use ≥ 16 seeds per cell (176+ episodes),
  ~25 min for all baselines.

**Follow-up: escalation by real backup workers** (decided with the user, replacing the
2 abstract slots):
- **Who:** for each work item an escalation does by hand, the 2 workers with the most
  sessions for it *outside* the top-3 pool (the next ranks after the pool's):
  - D1 → `W_Complete application`: User_10, User_61;
  - D2 → `W_Call after offers`: User_24, User_28;
  - D3 → `W_Call incomplete files`: User_29, User_28.
- **Mined:** who they are, what they are qualified for, and their attendance (share of days
  present). They work the normal shift and take only escalations.
- **Escalate is masked** while none of them is on shift and free, so it is unavailable at
  night and on Sundays.
- **Still assumed:**
  - that such a backup role exists, and its size of 2 per work item (swept in sensitivity);
  - the handling time, 0.5–2 working hours: a manual workaround is slower than the logged
    session (0.3–2.7 min median), and with logged times two workers could clear ~50
    cases an hour, so escalation would no longer be scarce;
  - the success probability, 0.95.
- **Do-nothing runs are unchanged** (the team's attendance has its own random stream), so
  the Loss_base calibration stays valid; checked bit-for-bit on D2 168 h.
- **Baseline benchmark rerun** (same 11 cells × 8 seeds). Policies that never escalate are
  unchanged:

  | Policy | Before (2 slots) | Now (backup workers) |
  |---|---|---|
  | do-nothing | 1.017 | 1.017 |
  | always substitute | 0.608 | 0.608 |
  | always escalate | 0.774 | 0.856 |
  | greedy | 0.748 | 0.671 |
  | rule | 0.583 | 0.672 |

  - **Escalation is scarcer now:** the backup workers are absent on some days and at
    night.
  - **The rule lost most on D2 168 h** (3.28 → 4.51, worse than do-nothing); not yet
    diagnosed.
  - **Best baseline is now always-substitute (0.61)**, then greedy and rule (0.67). On the
    robustness cells all are within noise of do-nothing (0.33–0.36).

### Phase 9 — Training 🔄
- **`train_loan_DRL.py`:** MaskablePPO with constant LR and entropy bonus.
- **Evaluation and selection:** 33-episode fixed-seed evaluation (3 per cell); keep every
  checkpoint and pick the final model on the benchmark seeds (200+).
- **`baselines_loan.py --agent NAME=PATH`** benchmarks trained models on the same episodes
  as the baselines.

**Run 1: process-wide reward** (`loan_ppo_w1_s0`, 500k steps, reward = change in the
process-wide Loss between two decisions, VecNormalize):
- Healthy training (entropy −1.12 → −0.85, explained variance 0.94–0.99, approx_kl ≈ 0.01),
  but the evaluation score swung between 0.57 and 0.89 without a trend.
- Benchmark (11 cells × 8 seeds): best checkpoint 0.811 (276k), against always-substitute
  0.608, greedy 0.671, rule 0.672, do-nothing 1.017. The checkpoint with the best
  evaluation score (0.566) scored 0.891: 3 episodes per cell are too few to select on.
- Learned the documents surge (D3 168/240 h: 0.30–0.60 vs do-nothing 0.98/1.39), not
  the long D1 outage (3.1–3.6 vs substitute 0.49): it retried instead of substituting.
- **Diagnosis:** the reward was badly attributed. ΔLoss between two decisions is the
  damage of all ~1,900 open cases, and a delay shows up weeks later, hundreds of
  decisions on; with gae_lambda 0.95 the credit fades within ~100 decisions. The cost of
  substituting was immediate and certain, its benefit invisible.

**Changes after run 1:**
- **Retry masked in D1/D2 outages.** A waiting case resumes the moment the service is
  restored, so retry could only resume it later and at a cost; it stays for missing
  documents (re-fetch). The rule baseline now waits in an outage unless the deadline is
  within 7 days (then escalate, or substitute without a free backup worker).
- **Per-case reward** (`RIMS_REWARD=case`, default; `global` reproduces run 1):
  - `QTracker.case_loss()`: at every instant the Loss rate is split equally among the
    off-track cases, so the shares add up to the Loss exactly (checked to 1e-14). In
    D1 168 h the cases that get decisions hold ~60% of the Loss, in D2 72 h / D3 ~21–26%.
  - An expired case is charged as if it stayed off track until the horizon, otherwise
    leaving the open set would make an expiry free.
  - r_i = −( w·N[e]·[(1−β)·ΔL_case,i + β·ΔLoss_i] / Loss_base[e] + κ_a ), β = 0.2:
    the case's own share up to its next decision or the horizon, plus a fifth of the
    process-wide term for the effects on other cases (reassign, scarce backup workers).
    Scaled by N[e] (per-decision units, a positive constant per scenario), no VecNormalize.
  - **`loan_ppo.CaseMaskablePPO`:** collects whole disruption episodes, regroups the
    steps into per-case chains (1.1–1.3 decisions long), recomputes values and
    log-probs with the current policy, then the usual GAE and PPO update.
  - The evaluation objective is unchanged, so runs are directly comparable; trained
    from scratch.

**Run 2: per-case reward, β = 0.2** (`loan_ppo_w1_s0_case_b0.2`, 100k steps): collapsed to
do-nothing (entropy −0.83 → −0.15, explained variance ≈ 0, value loss 26–100).
- **Cause:** the β part charged the process-wide Loss *until the next decision*. After the
  last decision of an episode (day 7 in D2 168 h) that is all the remaining Loss, up to
  day 50: one random case was charged ~1,100 against a normal 0–3. That outlier dominated
  the value targets and the normalised advantages, and only the certain cost κ remained.
- **The per-case part itself points the right way:** in D1 168 h a case costs 1.4–1.6
  when waiting and 0.31–0.34 when substituted; in D1 24 h ~0 vs 0.30. With β = 0 the
  largest per-case cost is ~19. A bounded process-wide term (e.g. a fixed 24 h window
  after the decision) is left for later.

**Environment v2** (decided with the user):
- **Every shock lasts 72–120 h** (uniform, hidden); short outages removed; the training
  cycle is one cell per scenario, length drawn.
- **D4 demand spike**, arrivals × 3 for the shock's length. Do-nothing sweep
  (`sweep_demand_spike.py`, 16 seeds, paired with the same episodes without a spike):

  | Factor | Extra applications | Loss (Q-days) | Lowest 24 h Q | Waits > 1 working day |
  |---|---|---|---|---|
  | 1 | 0 | 0.005 | 0.886 | 8 |
  | 2 | 327 | 0.035 | 0.877 | 13 |
  | 3 | 682 | 0.138 | 0.866 | 3–7 |
  | 5 | 1,370 | 0.56 | 0.828 | 50 |
  | 8 | 2,469 | 1.41 | 0.765 | 2,190 |

  At × 3 the pool absorbs the work; Q first *rises* (young cases), and the Loss comes at
  days ~30–45, when the spike's cohort reaches its deadlines together. So the planned
  "waits > 1 working day" trigger would fire ~5 times per episode.
- **D4 trigger (option A):** a spike-cohort case in `await_customer` or `incomplete`, on
  track, slack ≤ 48 h, once per stage visit, in working hours. Actions: reassign (calls
  first), substitute (written reminder: remaining customer waits × U[0.7, 0.9]),
  escalate (personal call by a `W_Call after offers` backup worker: × U[0.3, 0.6],
  success 0.95). Only cohort cases get shortenable waits, so D1–D3 are bit-for-bit
  unchanged (checked against their calibration).
- **Observation: 38 features** (+ scenario D4, cause overload, arrivals of the last 24 h,
  work waiting for a worker, spike-cohort flag).
- **Calibration (do-nothing, seeds 0–15):**

  | Scenario | Loss_base (Q-days) | sd | N decisions |
  |---|---|---|---|
  | D1 offer outage | 0.056 | 0.053 | 649 |
  | D2 telephony outage | 0.030 | 0.035 | 872 |
  | D3 documents surge | 0.016 | 0.014 | 276 |
  | D4 demand spike | 0.149 | 0.084 | 250 |

- **Training settings for run 3:** per-case reward with **β = 0** (now the default);
  evaluation on **8 episodes per scenario** (32, seeds 1000+), since 3 per cell picked
  lucky checkpoints in run 1.
- **Baselines on v2** (4 scenarios × 16 seeds, seeds 200+; lower is better):

  | Policy | Objective | vs do-nothing | D1 | D2 | D3 | D4 |
  |---|---|---|---|---|---|---|
  | do-nothing | 1.164 | — | 1.16 | 0.95 | 1.41 | 1.14 |
  | always retry | 0.954 | −0.21 ± 0.23 | 1.16 | 0.95 | 0.57 | 1.14 |
  | always reassign | 1.085 | −0.08 ± 0.23 | 1.16 | 0.95 | 0.96 | 1.26 |
  | always substitute | 0.967 | −0.20 ± 0.20 | 0.34 | 0.98 | 1.41 | 1.14 |
  | always escalate | 1.068 | −0.10 ± 0.23 | 0.95 | 0.97 | 1.23 | 1.13 |
  | greedy | 0.931 | −0.23 ± 0.29 | 0.23 | 0.84 | 1.47 | 1.18 |
  | **rule** | **0.814** | **−0.35 ± 0.27** | 0.37 | 1.30 | 0.57 | 1.01 |

  The best baseline per scenario would give ~0.66: room for an agent that adapts. D2
  has little room (written contact adds about as much delay as the rest of the outage);
  D4 barely responds to any action with the assumed effect sizes.
- **Note on noise:** an action that draws random numbers (handling time, effects) shifts
  the main random stream, so paired episodes diverge after the first action (e.g. a
  Saturday with no worker on shift, P = 0.03, in one run and not the other).

#### Run 3 (β = 0, 100k) and run 4 (windowed β = 0.5, 100k)
- **Run 3** (`loan_ppo_w1_s0_case_b0_100k`), benchmark on seeds 200+ × 16: best 0.944
  (92k), final 1.176 (do-nothing everywhere). Only D1 substitute learned; D3 retry
  dropped from 35% to 7%.
- **Diagnosis:** in D3 the cases with a decision carry only ~5% of the Loss; it comes at
  days 25–45 on other cases (arrived before the onset or after the surge), so the
  case-only reward cannot see retry's benefit. A 48 h window would hold none of it.
- **Windowed shared term** (commit 7fa0caf): the Loss at time s is split equally among
  the decisions made before s (window = whole horizon), weight β = 0.5. Shares sum to
  the Loss exactly; the largest single share ~1× the mean (worst step ~3.4 vs ~1,096
  with the old β term).
- **Run 4** (`loan_ppo_w1_s0_case_b0.5_100k`): best 0.951 (61k), final 0.993; D1 kept,
  D3 still do-nothing under the deterministic policy (retry 19% when sampled).
- **Marginal-benefit check** (treat 10/25/50% at random): the treated case's own benefit
  fell below κ between 10% and 25% in D1 (where the agent settled, ~20%) and was below κ
  at every rate in D3, while the objective kept improving up to ~50%. Half of all
  deadline misses cost the case nothing (charged only while Q < Q_ref, ~8% of the time).

#### Threshold-free Loss, new κ, D4 out of the cycle
- **Q_ref removed:** Loss = ∫ (1 − Q) dt; a case is charged dt/open for every off-track
  moment. Recovery (reported only) uses the normal mean Q, 0.901.
- **κ:** do_nothing 0, retry 0.05, reassign 0.15, substitute 0.20, escalate 0.30.
- **Normalisation:** normal operation accrues 4.68 ± 0.17 Q-days per episode, so each
  calibration episode is paired with its undisrupted twin (same seed and onset,
  zero-length disruption). objective = (Loss − Loss_normal)/Loss_base + Σκ/N.

  | Scenario | Loss_base = extra Loss (Q-days) | sd | N decisions |
  |---|---|---|---|
  | D1 offer outage | 0.173 | 0.138 | 649 |
  | D2 telephony outage | 0.168 | 0.219 | 872 |
  | D3 documents surge | 0.172 | 0.159 | 276 |
  | D4 demand spike | 0.031 | 0.207 | 250 |

- **D4 left out of the training cycle:** it adds no measurable Loss under this
  definition (the spike's new cases are on track and lift Q at first), and dividing by
  its tiny Loss_base made its objective pure noise.
- **Baselines** (D1–D3 × 16 seeds, seeds 200+):

  | Policy | Objective | vs do-nothing | D1 | D2 | D3 |
  |---|---|---|---|---|---|
  | do-nothing | 0.916 | — | 0.93 | 0.91 | 0.91 |
  | always retry | 0.817 | −0.10 ± 0.14 | 0.93 | 0.91 | 0.61 |
  | always reassign | 0.900 | −0.02 ± 0.15 | 0.93 | 0.91 | 0.86 |
  | always substitute | 0.821 | −0.09 ± 0.26 | 0.48 | 1.08 | 0.91 |
  | always escalate | 0.987 | +0.07 ± 0.29 | 1.21 | 0.96 | 0.80 |
  | greedy | 0.827 | −0.09 ± 0.27 | 1.06 | 0.63 | 0.79 |
  | **rule** | **0.595** | **−0.32 ± 0.29** | 0.43 | 0.93 | 0.43 |

  Best baseline per scenario ≈ 0.50.

#### Common random numbers and the do-nothing twin
- **Problem:** without Q_ref every case's normal lateness enters its reward (per-case
  reward ~ −21 against action effects ~1), and with one random stream two runs of the
  same seed diverge after the first action, so paired comparisons kept the full spread
  of normal operation (±0.17 Q-days per episode, about Loss_base).
- **Random streams:** separate streams for arrivals, attendance, the escalation team,
  and per case (per stage visit for routing/waits and for work durations; one for its
  action outcomes). Validation unchanged (9/9, KS 0.059); run time unchanged after
  creating per-case streams lazily and sampling choices by inverse CDF.
- **Do-nothing twin** (`loan_env`, `twin=True`): every episode is replayed with every
  decision do_nothing; both reward parts and the objective count the difference.
  Check: an all-do-nothing agent equals its twin exactly (Loss terms ~1e-14, objective
  1.000000; the baselines' inline do_nothing gives the same episode).
- **Own benefit per treated case** (β = 0, treat 10/25/50% at random): D1 substitute
  1.72 / 1.42 / 1.20 (κ 0.20), D3 retry 1.18 / 1.00 / 1.03 (κ 0.05), D2 substitute
  ~0.14 ± 0.21 (κ 0.20). Untreated D3 cases: 0.004 ± 0.14. Before: D3 retry 0.01–0.04.
- **Recalibration** (paired with the undisrupted twin): Loss_base D1 0.178, D2 0.152,
  D3 0.110 Q-days; spread of the extra Loss 0.105 / 0.091 / 0.045 (was 0.14 / 0.22 / 0.16).
- **Baselines** (paired objective 1 + (Loss − Loss_do-nothing)/Loss_base + cost; D1–D3 ×
  16 seeds, seeds 200+); intervals 3–6× tighter than before:

  | Policy | Objective | vs do-nothing | D1 | D2 | D3 |
  |---|---|---|---|---|---|
  | do-nothing | 1.000 | — | 1.00 | 1.00 | 1.00 |
  | always retry | 0.758 | −0.24 ± 0.11 | 1.00 | 1.00 | 0.27 |
  | always reassign | 0.923 | −0.08 ± 0.05 | 1.00 | 1.00 | 0.77 |
  | always substitute | 0.772 | −0.23 ± 0.14 | 0.24 | 1.08 | 1.00 |
  | always escalate | 0.872 | −0.13 ± 0.05 | 0.92 | 0.98 | 0.72 |
  | greedy | 0.785 | −0.22 ± 0.07 | 0.57 | 0.99 | 0.79 |
  | **rule** | **0.371** | **−0.63 ± 0.17** | 0.26 | 0.97 | −0.11 |

  Below 0 (rule in D3): the episode ends with less lateness than without the
  disruption, since the actions also spare cases the documents loop they would have
  gone through anyway.

#### Runs with the twin (100k + 100k) and the escalation shadow price
- **Run `loan_ppo_w1_s0_case_b0.5_twin_100k`** (benchmark seeds 200+ × 16, paired
  objective): 0.394 at 61k, flat to 200k (0.394–0.398). Against the rule (7 d):
  D1 tie, D2 −0.17 ± 0.08 (agent better), D3 +0.26 ± 0.12 (rule better), overall tie.
- **Rule threshold sweep** (`rule_<days>d`): 1 d 0.359, 2 d 0.360, 4 d 0.370, 7 d 0.371,
  10 d 0.413, 14 d 0.465, 21 d 0.572; best per scenario 0.355. The agent ties the
  tuned rule as well.
- **D3 diagnosis:** not reassign (for urgent cases reassign is worse than retry in
  reward and policy). The agent escalates every case whenever a backup worker is
  free, at any slack: escalate beats retry in each case's own reward (+0.35 even at
  14+ days of slack), but the worker is then busy when an urgent case arrives.
  "Escalate only if urgent, else retry" scores −0.09 against the agent's 0.18 (seeds 300+).
- **Fix: shadow price of the backup workers**, measured from the process
  (`measure_shadow_price.py`): extra Loss per escalation forgone with one backup
  worker fewer, under the current agent. D1 0 (it hardly escalates), D2 0 (no
  measurable cost), D3 1.19 per escalation. `loan_env` charges (1 − β) × price to
  every escalation, reward only. With it, escalate − retry in D3: +2.90 at < 2 days
  slack, ~0 at 2–14 days, −0.25 at 14+ days.

- **Continuation from 200k with the price** (`…_twin_shadow`, to ~304k): D1/D2 slightly
  better (0.386–0.391 overall), D3 unchanged (0.149). The near-deterministic policy
  kept escalating whenever a backup worker was free; the new signal (≈0 at 2–14 days of
  slack, −0.25 ± 0.34 beyond) was too weak to break the habit.
- **Fresh run with the price** (`loan_ppo_w1_s0_case_b0.5_shadow_fresh_100k`): learned
  an urgency line by itself. It escalates when a backup worker is free in 99% of
  overdue cases, 88% at 2–4 days, 70% at 4–7, 21% at 7–14 and 0% beyond 14 days, so
  the backup workers are free ~45–50% of the time (was 14–16%). Benchmark: 0.436 (31k)
  → 0.327 → 0.318 → **0.303** (final; D1 0.226, D2 0.792, D3 −0.108). Against the
  rule (7 d): **−0.067 ± 0.051** (D1 −0.03 ± 0.04, D2 −0.17 ± 0.10, D3 +0.00 ± 0.10).
  Against the best-tuned rule (1 d): **−0.055 ± 0.048**. First agent to beat the rule
  significantly; still improving at 100k.
- **Shadow price re-measured** under the fresh agent: D1 0, D2 1.34 (0.35–2.64),
  D3 3.54 (1.29–5.85). It is about 3× higher in D3, since escalations now go to cases
  that need them. Stored for the final runs.

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
