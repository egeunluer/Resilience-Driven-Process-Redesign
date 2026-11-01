# Stage-model config — schema and semantics

The simulator (`loan_model/simulator.py`, Phase 3) is a generic engine. Everything
specific to one event log lives in two JSON files in `Module 1/artifacts/`:

| File | Written by | Content |
|---|---|---|
| `<log>_activity_map.json` | hand (log-specific input) | which activities mark stage entries, which work item belongs to each stage, timers, flags |
| `<log>_stage_model.json` | `loan_model/mine_stage_model.py` | everything mined: routing, waits, work sessions, workers, calendar, arrivals, deadlines, validation targets |

To use another log (e.g. BPI2012), write a new activity map and rerun the miner. No code
changes are needed as long as the process fits the stage semantics below.

## Stage semantics (how the simulator reads the model)

A case is a sequence of **stage visits**. A visit starts at a stage-entry milestone
(e.g. `A_Validating`) and ends at the next one. In the activity map, an entry key is
either an activity name (any lifecycle) or `ACTIVITY@lifecycle`, for logs that mark a
stage change only by a work item's lifecycle event (BPI2012:
`W_Valideren aanvraag@schedule`). Each stage has at most one **work item**
(e.g. `W_Validate application`), done in one or more **sessions** by qualified workers.
Between sessions the work item is suspended: the case waits without using a resource
(customer response, documents, internal hand-over).

On entering stage `s` (visit number `v`, with case flags `f`):

1. **Exit:** draw the next stage `n` from `stages[s].exits_by_visit[bucket(v)]` (or
   `exits_by_flag` when a flag such as `fraud` is set). End stages finish the case.
2. **Optional work items:** draw them with their probability per visit (e.g. fraud
   check in validation); they set flags.
3. **Visit pattern:** draw it for the pair `(s, n)` from `stages[s].transitions[n]`:
   - `n_sessions`: empirical distribution of the number of human work sessions;
   - `initial_delay`: entry → first session start (quantiles, s);
   - `gap`: session end → next session start (quantiles, s), resource-free;
   - `tail`: last session end → exit (quantiles, s), resource-free;
   - with 0 sessions, the whole visit is one resource-free wait: `visit_duration`.
   - `initial_delay_w`, `gap_w`, `tail_w`: the same waits in **working time** (only the
     calendar's working days and hours count, `loan_model/calendar_time.py`). The
     simulator uses these when present. A wait that ends in a work session always ends
     inside the shift in the log; replaying it in wall-clock time from a random moment
     would often end at night and queue until morning, counting off-hours twice.
4. **Sessions:** each session needs one free worker qualified for the work item. It
   lasts that worker's session duration (`workers[w].sessions[item]`, quantiles, s),
   with a fallback to `work_items[item].session_duration`. Workers follow the calendar
   and their `active_day_share`.
5. **Timers:** if `timers` names the stage, the visit races against the timer (e.g. the
   offer expiry, 30 days after the last offer sent). Whichever comes first decides the exit.

All durations are in seconds. Quantile arrays have `n_quantiles` points (0 %, 1 %, …,
100 %) and are sampled by inverse CDF with linear interpolation.

## Top-level keys of `<log>_stage_model.json`

| Key | Content |
|---|---|
| `meta` | log, mapping file, time zone, date span, number of cases, mining date |
| `arrivals` | `rates_per_hour_of_week` (168, local time, Monday 00:00 = slot 0), `per_day` |
| `case_flags` | share of cases per flag (e.g. `online`) |
| `case_attributes` | quantiles / shares of `RequestedAmount`, `ApplicationType`, `LoanGoal` |
| `offers` | offers per case, per stage visit, online-only channel share |
| `stages` | per stage: `work_item`, `visits`, `exits_by_visit`, `exits_by_flag`, `transitions`, `optional_work_items` |
| `work_items` | per work item: global session-duration quantiles, sessions per visit, qualified workers |
| `workers` | per human worker: `active_day_share`, per work item `n` and session quantiles |
| `calendar` | working days, `hour_min`, `hour_max` (local time), share of sessions on weekends, `session_share_by_weekday` (scales daily presence) |
| `timers` | per timer: anchor, target, mined delay quantiles, share of target exits it explains |
| `deadlines` | `D_k` per stage: p90 of case age at stage exit (for Q(t)) |
| `validation_targets` | cycle time quantiles, outcome shares, incomplete share, validation rounds, offers per case, open cases, arrivals/day |
