# Assumptions Table — White-Box Resilience Engine (BPIC2017, full log)

Every parameter of the simulator, the disruption layer, the recovery actions and the
reward, with its value or range and where it comes from. Kept up to date as the
`new-approach` phases are implemented (see `drl_resilience_design_decisions.md` §13/§15).

**Source** — *mined*: computed from `Event Logs/BPI Challenge 2017.xes`; *imposed*:
chosen by us but derived from mined data; *literature*: taken from cited work;
*assumed*: our choice, always swept in the sensitivity analysis.

Status: section 1 is mined (Phase 2). Sections 3–4 get their ranges in Phases 5–6.

## 1. Base process (mined)

All values below are written by `loan_model/mine_stage_model.py` into
`Module 1/artifacts/bpic2017_stage_model.json` (mapping: `bpic2017_activity_map.json`).
Local time zone Europe/Amsterdam (the log stores UTC).

| Parameter | Value | Source | Rationale / note |
|---|---|---|---|
| Cases | 31,509 (2016-01-01 – 2017-02-01) | mined | full log, all A_/O_/W_ events |
| Arrival rate | 86.2 cases/day, 168 hour-of-week rates | mined | first event per case; `weekly_rates` from `mine_arrival_profile.py` |
| Online share | 0.648 | mined | `A_Create Application` by `User_1` |
| Stages | intake → complete → await_customer → validation ⇄ incomplete → accepted / denied / cancelled | mined (mapping by hand) | entry milestones `A_Create Application`, `A_Concept`, `A_Complete`, `A_Validating`, `A_Incomplete`, `A_Pending` / `A_Denied` / `A_Cancelled` |
| Routing | per stage and visit number (1, 2, 3+), e.g. await_customer → validation 0.699 / cancelled 0.297; validation (1st) → incomplete 0.686 / accepted 0.209 / denied 0.102 | mined | `exits_by_visit`; fraud-flagged validations route separately (55% denied) |
| Work | active sessions (start/resume → suspend/complete/abort) per worker and work item, e.g. `W_Validate application` median 2.1 min, p90 17.9 min | mined | **not** Module 1's start→complete pairing, which includes days of suspension in this log |
| Resource-free waits | per (stage, next stage): initial delay, gaps between sessions, tail | mined | e.g. customer return: await_customer → validation visit median 7.8 d, p90 18.0 d |
| Offers per case | 1: 72.8%, 2: 20.9%, 3: 4.3%, 4+: 2.0% | mined | `O_Create Offer` |
| Offer channel | mail+online 95.1%, online only 4.9% | mined | alternative channel used by *substitute* (D1) |
| Offer expiry timer | 30.7 d after the last offer sent (p10–p90 30.5–30.9) | mined | explains 85% of await_customer → cancelled exits; **BPI2012 shows the same 30.7-day rule** |
| Fraud check | 0.62% of validation visits | mined | `W_Assess potential fraud`, 34 qualified workers |
| Workers | 145 human workers; qualified per work item: 78–124; mean active-day share 0.28 (≈ 40 active per working day) | mined | ≥ 3 sessions per work item to count as qualified |
| Calendar | Mon–Sat, 07:00–21:00 local; 6.8% of sessions on weekends; daily presence scaled by weekday share | mined | 1st / 99th percentile of session start hours |
| Wait measurement | waits ending in a session measured and replayed in working time | imposed | avoids counting off-hours twice (Phase 3 validation: cycle-time median +14% without it) |
| Dispatcher | fastest free qualified worker, FIFO per work item | imposed | not a decision of the agent; strongest allocation rule found on Consulta |
| Warm-up | 40 days (95% of steady-state open cases by day 35) | imposed | ~1.1 s per episode |
| Final outcome | accepted 54.7%, cancelled 33.1%, denied 11.9% | mined | validation target |
| Cycle time | median 19.1 d, p90 35.0 d | mined | validation target |
| Open cases | median 1,937 (p10 1,507, p90 2,343) | mined | daily sample; validation target and warm-up length |

## 2. Resilience measure (imposed, grounded)

| Parameter | Value | Source | Rationale / note |
|---|---|---|---|
| Stage deadline `D_k` | p90 of case age when leaving stage k, floor 24 h: intake 1 d, complete 3.6 d, await_customer 32.8 d, validation 28.4 d, incomplete 35.5 d | imposed (from mined ages) | "on track" if age ≤ `D_k`; the floor only changes intake (7.8 min) |
| Q(t) | share of open cases on track | imposed | leading indicator, reacts at once, uses all open cases |
| Q_mean | 0.904 (daily sd 0.009) in undisrupted runs | imposed | calibrated, seeds 0–15 × 120 days |
| Q_ref | removed | — | was the 5th percentile of daily-mean Q (0.886, top-3 pool) and the Loss threshold; it made about half of all deadline misses free (a case was charged only while Q < Q_ref, ~8% of the time in D1–D3), so the agent saw little own benefit from acting |
| Resource pool | working default: top 3 workers per work item (16 workers, 36% utilised) | imposed | constrained pool; still passes validation (9/9) |
| Loss | ∫ (1 − Q(t)) dt, no threshold | literature | recovery triangle (corpus P43, P35); every off-track moment counts. Normal operation accrues ~0.099 Q-day per day (4.68 ± 0.17 Q-days per 50-day episode) |
| Hard penalty | expiries caused by delay (timer wins the race) | imposed | baseline 1.4/day; customers who never respond are not penalised |
| Loss_base[e], Loss_normal[e] | Loss_base: mean extra Loss of a do-nothing episode over its undisrupted twin (same seed and onset, zero-length disruption): D1 0.178, D2 0.152, D3 0.110 Q-days (sd 0.105, 0.091, 0.045). Loss_normal: the twin's mean Loss, 4.64 Q-days. 1 Q-day ≈ 1,900 case-days late | imposed (calibrated, seeds 0–15) | objective = 1 + (Loss − Loss of do-nothing on the same episode)/Loss_base + Σκ/N; do-nothing = 1 exactly |
| Random streams | separate streams for arrivals, attendance, the escalation team's attendance, and per case: per stage visit (routing and waits; work durations) and for its action outcomes | imposed | common random numbers: an action changes no other case's draws, so runs of one seed differ only by the actions (validation unchanged, 9/9) |
| Recovery | disruption over, all blocked steps done, 24 h rolling Q ≥ normal mean Q (0.901) | imposed | reported only, not part of the Loss or the reward |
| Episode horizon | fixed 50 days after onset | imposed | late harm arrives by day ~40 (D2); recovery time reported separately |
| Customer patience β | not implemented (deferred option) | assumed | P(cancel) = 1 − exp(−β · disruption delay); β = 0 is the current model |

## 3. Disruption layer (assumed, swept)

| ID | Parameter | Range | Source | Note |
|---|---|---|---|---|
| all | shock length | uniform 72–120 h for every shock type, drawn per episode | assumed | hidden from the agent; outages up to ~72 h did about as much harm as normal noise, so short shocks were removed |
| D1 | offer dispatch outage | blocks `complete → await_customer` | assumed | one outage per episode |
| D2 | telephony outage | blocks `W_Call after offers`, `W_Call incomplete files` | assumed | one outage per episode |
| D3 | incomplete-documents surge `m` | level 2: m 0.5 (level 1: 0.25) | assumed | `p_inc × (1 + m)` capped at 0.95, base `p_inc` mined |
| D3 | portal upload delay added to the incomplete loop | level 1: 24–96 h; level 2: 72–168 h | assumed | makes D3 harmful (without it Loss ≈ 0.006: base `p_inc` is already 0.69 on the first round) |
| training cycle | one cell per scenario, D1–D3, length drawn | — | assumed | replaced the fixed-length cycle (8–240 h). D4 left out: under the threshold-free Loss it adds 0.03 ± 0.21 Q-days per episode, no measurable harm |
| D4 | demand spike: arrival rate × 3 (hours that start during the spike) | 3 | assumed | do-nothing sweep (`sweep_demand_spike.py`): the pool absorbs ×3 (almost no work item waits a working day); Loss 0.14 Q-days, arriving at days ~30–45 when the spike's cohort reaches its deadlines together; queues only blow up around ×8 |
| D4 | decision trigger | spike-cohort case in `await_customer` / `incomplete`, on track, slack ≤ 48 h, once per stage visit, working hours | assumed | targets where the spike's harm is; ~250 decisions per episode |
| D4 | reminder effect (substitute) | remaining customer waits of the stage visit × U[0.7, 0.9] | assumed | automated written reminder, no worker |
| D4 | personal call effect (escalate) | remaining customer waits × U[0.3, 0.6], success 0.95, backup workers of `W_Call after offers` | assumed | reminder calls exist in the log (`W_Call after offers`); their effect size is not identifiable from it |
| D4 | reassign | the case's remaining calls in the stage go to the front of the queue | assumed | small effect at ×3 (short queues) |
| — | staff absence (former D4) | TBD | assumed | later |
| D5 | fraud-flag surge factor | TBD | assumed | base rate mined |
| D6 | SLA-risk alert threshold (slack) | TBD | assumed | expiry clock is mined |

## 4. Recovery actions and costs (assumed, swept)

| Action | Cost κ (× c_p) | Success / effect | Source | Note |
|---|---|---|---|---|
| do-nothing | 0 | waits for natural resolution | — | |
| retry | 0.05 | missing documents only: automatic re-fetch after 1 working h, finds them with p 0.4; at most 2 per case. Not offered in D1/D2 outages | assumed | ordering: literature (Russell 2006, Lerner 2010, P20). Masked in outages because a waiting case resumes the moment the service is restored, so retrying can only resume it later and at a cost |
| reassign | 0.15 | D3: skips the portal delay, calls to the front of the queue | assumed | delays other cases (coupling) |
| substitute | 0.20 | alternative channel; rework prob r_sub | assumed | D1: online-only send (channel mined) |
| escalate | 0.30 | q_esc ≈ 0.95; handling 0.5–2 working h | assumed | manual workaround, slower than the logged session (0.3–2.7 min median) |
| escalation backup team | 2 workers per escalated work item, ranked just outside the top-k pool | `W_Complete application`: User_10, User_61; `W_Call after offers`: User_24, User_28; `W_Call incomplete files`: User_29, User_28 | mined (who, qualification, attendance); assumed (that this role exists, size 2) | escalations only; masked when none on shift and free; size swept in sensitivity |
| c_p (reference penalty) | 1 (relative) | — | literature | Metzger et al. 2020: relative costs, κ swept 0.1–0.5 |
| Loss_base, N (reward normalisation) | per scenario, over its cycle cells (see section 2) | — | calibrated | |
| per-case reward | r_i = −[κ_a + (N/Loss_base)·((1−β)·ΔL_case,i + β·S_i)], both Loss parts minus the same quantity in the do-nothing twin (the episode replayed with every decision do_nothing), so lateness that happens anyway cancels; ΔL_case: the case's own Loss (dt/open for every off-track moment, and after an expiry) until its next decision; S_i: windowed share of the process Loss (the Loss at time s split equally among the decisions made before s, window = whole horizon) | — | imposed | β = 0.5. β = 0 slid to do-nothing (the benefit of D3 retry falls on other cases); the earlier β term (Loss until the next decision) gave the last decision all remaining Loss and collapsed training |
| w (resilience vs cost weight) | 1 (default) | — | assumed | Pareto sweep |
| expiry / deadline-miss penalty | TBD × c_p | — | assumed | residual penalty |
