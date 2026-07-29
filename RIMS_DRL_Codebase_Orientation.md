# Orientation: `RIMS_DRL-main` (Modules 2-3 codebase)

Quick-start guide for inspecting `Modules 2-3/RIMS_DRL-main/` — where it fits in the thesis, and
what order to read things in.

---

## Where it is and what it is

`Resilience-Driven-Process-Redesign/Modules 2-3/RIMS_DRL-main/`

It's the vendored fork of Meneghello/Middelhuis's **RIMS_DRL** (the BPM 2024 paper repo — hybrid
DES + DRL for resource allocation). Per the TDR, this is where Modules 2 (Hybrid Simulation) and 3
(DRL Environment/Agent) live. Important: **this is no longer the stock upstream repo** — it has
already been substantially modified in place for the thesis (shocks, resilience rewards, empirical
durations all wired in).

---

## Best starting point: two docs one level up, not the code itself

Before touching the ~3,500 lines of Python, read these two files sitting in `Modules 2-3/`
alongside the codebase — they're purpose-written for exactly this orientation task:

1. **`RIMS_DRL_Agent_Briefing.md`** — the *why*: what the thesis needs from this repo, what to
   inherit from stock RIMS_DRL vs. replace, and the full list of design decisions (D1 duration
   swap, D5 shocks, D6/D9 reward functions, D7 observation extension, D10 calendars, etc.) with
   rationale.
2. **`RIMS_DRL_Modification_Plan.md`** — the *what actually happened*: a step-by-step changelog
   (Steps 1–34, organized into 14 phases) of every edit made to the original repo, ending in an
   "Execution order summary" table mapping each step to the file it touched. This is effectively a
   diff-with-commentary against upstream.

Reading these first means you understand the modifications as a narrative instead of
reverse-engineering them from a diff.

---

## Then, reading order through the code itself

Once you have that context, a sensible dependency-order pass through the code:

- **`train_DRL.py`** — short (203 lines) and the best entry point: it constructs the environment,
  wires shock config, calibrates `CPL_base`/`phi_fix`, and kicks off `MaskablePPO.learn()`. Reading
  it top to bottom tells you how every other module gets invoked.
- **`gym_env.py`** (525 lines, the largest file) — the center of gravity. Observation/action space,
  action masking, both reward functions (RF1/RF2), shock wiring, and the cluster-layer
  `get_cluster_obs()` used for Module 4 all live here.
- **`process.py`** → **`event_trace.py`** — the SimPy simulation core (`SimulationProcess`,
  `Token`) and the point where activity duration is now looked up from the Module 1 empirical
  artifact instead of the original LSTM.
- **`shock.py`** — `ShockConfig`/`ShockState`/`shock_scheduler`, the resilience-specific addition.
- **`role_simulator.py`**, **`inter_trigger_timer.py`** — resource-as-SimPy-resource wrapper and
  the arrival process (with the demand-spike multiplier hook).
- **`evaluate_DRL.py`** — the FIFO/SPT/Random baselines used for comparison.
- **`collect_behavioral_log.py`** — the post-training script that feeds Module 4 (tier-level
  behavioral log).

---

## Current status (as of 2026-07-29)

All 34 steps in `RIMS_DRL_Modification_Plan.md` are done: LSTM/Prophet removed, empirical
durations wired in, shocks implemented, RF1 and RF2 both implemented and validated on a BPI2017
manual-stepping pilot, and the Module 4 behavioral-log collector already built. Module 1 artifacts
exist for all 4 target logs (bpi2017, bpi12w, helpdesk, consulta).

**One known open issue**, documented in Phase 14 of the Modification Plan: `next_decision_moment()`
in `gym_env.py` has no bound on how long it searches for a valid decision. With shocks enabled, a
`RESOURCE_REMOVAL` shock can disable enough of an activity's (already finite) resource pool that the
search stalls for real minutes before self-resolving — confirmed at both 25 and 150 traces.
Independent of `postpone` (removed entirely in Step 26). **Needs a fix (iteration/time bound +
fallback) before shock-enabled training runs can be trusted at scale.** This is the most likely next
concrete task in Modules 2-3.