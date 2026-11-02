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
| Q_ref | 0.890 (full pool) / 0.886 (top-3 pool) = 5th percentile of daily-mean Q without disruption | imposed | Loss and recovery threshold; below the mean, normal noise would count as Loss |
| Resource pool | working default: top 3 workers per work item (16 workers, 36% utilised) | imposed | constrained pool; still passes validation (9/9) |
| Loss | ∫ max(0, Q_ref − Q(t)) / Q_ref dt | literature | recovery triangle (corpus P43, P35); ~12 s/day without disruption |
| Hard penalty | expiries caused by delay (timer wins the race) | imposed | baseline 1.4/day; customers who never respond are not penalised |
| Loss_base[e] | do-nothing Loss per scenario | imposed | normalisation, like CPL_base |
| Recovery | disruption over, all blocked steps done, 24 h rolling Q ≥ Q_ref | imposed | 3-day full work stop: recovery after 4.2 d |
| Episode horizon | fixed 50 days after onset | imposed | late harm arrives by day ~40 (D2); recovery time reported separately |
| Customer patience β | not implemented (deferred option) | assumed | P(cancel) = 1 − exp(−β · disruption delay); β = 0 is the current model |

## 3. Disruption layer (assumed, swept)

| ID | Parameter | Range | Source | Note |
|---|---|---|---|---|
| D1 | offer dispatch outage: rate, MTTR | TBD | assumed | uncertain repair time |
| D2 | telephony outage: rate, MTTR | TBD | assumed | affects `W_Call after offers`, `W_Call incomplete files` |
| D3 | incomplete-documents surge `m`, duration | level 1: m 0.25, 72–168 h; level 2: m 0.5, 120–240 h | assumed | `p_inc × (1 + m)` capped at 0.95, base `p_inc` mined |
| D3 | portal upload delay added to the incomplete loop | level 1: 24–96 h; level 2: 72–168 h | assumed | makes D3 harmful (without it Loss ≈ 0.006: base `p_inc` is already 0.69 on the first round) |
| D4 | absent share of team, absence length | TBD | assumed | added after D1–D3, D6 |
| D5 | fraud-flag surge factor | TBD | assumed | base rate mined |
| D6 | SLA-risk alert threshold (slack) | TBD | assumed | expiry clock is mined |

## 4. Recovery actions and costs (assumed, swept)

| Action | Cost κ (× c_p) | Success / effect | Source | Note |
|---|---|---|---|---|
| do-nothing | 0 | waits for natural resolution | — | |
| retry | 0.05 | succeeds only once the outage is over; documents re-fetch 0.4; at most 2 per case | assumed | ordering: literature (Russell 2006, Lerner 2010, P20); cap keeps episodes short |
| reassign | 0.15 | D3: skips the portal delay, calls to the front of the queue | assumed | delays other cases (coupling) |
| substitute | 0.30 | alternative channel; rework prob r_sub | assumed | D1: online-only send (channel mined) |
| escalate | 0.50 | q_esc ≈ 0.95; senior pool capacity 1–3 | assumed | masked when no slot free; size vs fraud team |
| c_p (reference penalty) | 1 (relative) | — | literature | Metzger et al. 2020: relative costs, κ swept 0.1–0.5 |
| w (resilience vs cost weight) | TBD | — | assumed | Pareto sweep |
| expiry / deadline-miss penalty | TBD × c_p | — | assumed | residual penalty |
