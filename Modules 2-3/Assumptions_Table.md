# Assumptions Table — White-Box Resilience Engine (BPIC2017, full log)

Every parameter of the simulator, the disruption layer, the recovery actions and the
reward, with its value or range and where it comes from. Kept up to date as the
`new-approach` phases are implemented (see `drl_resilience_design_decisions.md` §13/§15).

**Source** — *mined*: computed from `Event Logs/BPI Challenge 2017.xes`; *imposed*:
chosen by us but derived from mined data; *literature*: taken from cited work;
*assumed*: our choice, always swept in the sensitivity analysis.

Status: values marked *(prelim.)* come from the first profiling pass and will be
replaced by the output of `loan_model/mine_bpic2017.py` (Phase 2).

## 1. Base process (mined)

| Parameter | Value | Source | Rationale / note |
|---|---|---|---|
| Cases | 31,509 (2016-01-01 – 2017-02-01) | mined | full log, all A_/O_/W_ events |
| Arrival rate | 86.3 cases/day *(prelim.)*, hour-of-week profile | mined | from `A_Create Application` timestamps |
| Online share | 0.65 *(prelim.)* | mined | applications created by `User_1`; all go through `W_Handle leads` |
| Offers per case | 1: 73%, 2: 21%, 3: 4%, 4+: 2% *(prelim.)* | mined | `O_Create Offer` count per case |
| Offer channel | mail+online vs online only (`O_Sent (online only)` 5.7% of cases) | mined | alternative channel used by *substitute* (D1) |
| Customer response delay (offer sent → returned) | median 7.8 d, p90 18.6 d *(prelim.)* | mined | resource-free wait, empirical quantiles |
| Offer expiry | 30 days after last offer sent | mined | 76% of `A_Cancelled` are by `User_1`, 30.7 d (p10–p90 30.5–30.9) after last `O_Sent` |
| Incomplete documents per validation round | 23,055 / 38,816 rounds *(prelim.)* | mined | `A_Incomplete` → `W_Call incomplete files` → `A_Validating` loop |
| Documents back after incomplete | median 1.0 d, p90 7.0 d *(prelim.)* | mined | `A_Incomplete` → next `A_Validating` |
| Fraud check | 1.0% of cases, 38 workers, mean 143 min work | mined | `W_Assess potential fraud`; outcome mostly Denied |
| Final outcome | Pending 54.7%, Cancelled 33.1%, Denied 11.9% | mined | last of A_Pending / A_Cancelled / A_Denied |
| W_ work durations | per (resource, activity), minutes | mined | Module 1 (`bpi2017.json`), true start/complete pairs |
| Workforce | ~145 workers, ~38 active per weekday, weekdays only | mined | qualified pairs from Module 1; calendars |
| Cycle time | median 19.1 d, p90 35.1 d | mined | validation target for the base simulator |
| Open cases | ~1,900 (1,329–2,486) | mined | validation target, warm-up length |

## 2. Resilience measure (imposed, grounded)

| Parameter | Value | Source | Rationale / note |
|---|---|---|---|
| Stage deadline `D_k` | p90 of case age when leaving stage k | imposed | "on track" if age ≤ `D_k` of current stage |
| Q(t) | share of open cases on track | imposed | leading indicator, reacts at once, uses all open cases |
| Q_base | mean Q(t) in undisrupted runs (≈0.9 by construction) | imposed | calibrated, seeds 0–15 |
| Loss | ∫ max(0, Q_base − Q(t)) / Q_base dt | literature | recovery triangle (corpus P43, P35) |
| Loss_base[e] | do-nothing Loss per scenario | imposed | normalisation, like CPL_base |
| Recovery tolerance | Q (1 working-day mean) ≥ Q_base − tol, tol TBD | assumed | swept |
| Episode horizon cap | 60 days after onset | assumed | |

## 3. Disruption layer (assumed, swept)

| ID | Parameter | Range | Source | Note |
|---|---|---|---|---|
| D1 | offer dispatch outage: rate, MTTR | TBD | assumed | uncertain repair time |
| D2 | telephony outage: rate, MTTR | TBD | assumed | affects `W_Call after offers`, `W_Call incomplete files` |
| D3 | incomplete-documents surge `m` | TBD | assumed | `p_inc × (1 + m)`, base `p_inc` mined |
| D4 | absent share of team, absence length | TBD | assumed | added after D1–D3, D6 |
| D5 | fraud-flag surge factor | TBD | assumed | base rate mined |
| D6 | SLA-risk alert threshold (slack) | TBD | assumed | expiry clock is mined |

## 4. Recovery actions and costs (assumed, swept)

| Action | Cost κ (× c_p) | Success / effect | Source | Note |
|---|---|---|---|---|
| do-nothing | 0 | waits for natural resolution | — | |
| retry | 0.05 | q_retry ≈ 0 during hard outage, ~0.6 transient | assumed | ordering: literature (Russell 2006, Lerner 2010, P20) |
| reassign | 0.15 | front of queue, fastest free qualified worker | assumed | delays other cases (coupling) |
| substitute | 0.30 | alternative channel; rework prob r_sub | assumed | D1: online-only send (channel mined) |
| escalate | 0.50 | q_esc ≈ 0.95; senior pool capacity 1–3 | assumed | masked when no slot free; size vs fraud team |
| c_p (reference penalty) | 1 (relative) | — | literature | Metzger et al. 2020: relative costs, κ swept 0.1–0.5 |
| w (resilience vs cost weight) | TBD | — | assumed | Pareto sweep |
| expiry / deadline-miss penalty | TBD × c_p | — | assumed | residual penalty |
