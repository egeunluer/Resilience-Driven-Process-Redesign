"""
Shock injection layer for the resilience DRL environment (D5).

Two shock types:
  RESOURCE_REMOVAL  — a fraction of a named resource group becomes unavailable.
                      Busy resources finish their current task; no pre-emption.
  DEMAND_SPIKE      — case arrival rate is multiplied by (1 + magnitude) for
                      the shock duration.

Onset times follow a homogeneous Poisson process in train mode (random,
overlaps allowed). In eval mode exactly one shock fires at a fixed time with
no overlap, giving a clean before/during/after recovery curve.
"""

from __future__ import annotations

import random
import math
from dataclasses import dataclass, field
from typing import Optional

# Must match gym_env.PLACEHOLDER_RESOURCE (duplicated, not imported, to avoid a
# circular import — gym_env imports from this module, not the reverse).
PLACEHOLDER_RESOURCE = '__NO_RESOURCE__'


# ---------------------------------------------------------------------------
# Configuration (set once at startup)
# ---------------------------------------------------------------------------

@dataclass
class ShockConfig:
    """
    Immutable configuration for the shock layer.

    Train mode parameters
    ---------------------
    lambda_rate     : float
        Poisson rate (shocks per unit sim-time). Inter-shock intervals are
        drawn from Exponential(1 / lambda_rate).
    shock_types     : list[str]
        Which shock types to sample from. Subset of {'RESOURCE_REMOVAL',
        'DEMAND_SPIKE'}.
    magnitude_range : tuple[float, float]
        (min, max) fractional rate increase for DEMAND_SPIKE only (so 0.5 →
        arrival rate × 1.5; the calibrated/eval regime uses 5-8, i.e. a
        6x-9x multiplier via 1+magnitude). NOT used for RESOURCE_REMOVAL —
        see resource_removal_magnitude_range. These two were incorrectly
        sharing this one field until the bug below was found and fixed:
        RESOURCE_REMOVAL's magnitude means "fraction of the eligible pool to
        remove" (must stay in [0,1]), so drawing from DEMAND_SPIKE's
        validated 5-8 range clamped every RESOURCE_REMOVAL shock to removing
        100% of an activity's eligible resources — stranding any token
        already pending for that activity with zero eligible resources for
        the rest of the episode, which crashed training
        (IndexError in gym_env.step(), found via a Consulta training smoke
        test). RESOURCE_REMOVAL was disabled for training until this field
        was split out; see resource_removal_magnitude_range below.
    resource_removal_magnitude_range : tuple[float, float]
        (min, max) fraction of the eligible resource pool to remove, for
        RESOURCE_REMOVAL only. Must stay within [0, 1] — magnitude_range's
        5-8 (valid for DEMAND_SPIKE's different multiplier semantics) would
        clamp to removing the entire pool if used here. Defaults to (0.2,
        0.5), the value this project has always used for RESOURCE_REMOVAL
        specifically (unrelated to DEMAND_SPIKE's own calibration history).
    duration_range  : tuple[float, float]
        (min, max) shock duration in simulation seconds. Shared by both
        shock types -- duration has the same units/meaning regardless of
        shock type, unlike magnitude.
    cycle_schedule  : list[tuple[str, float]]
        Optional ordered list of (shock_type, magnitude) pairs. If non-empty,
        fire_single_shock walks it in order -- one entry per episode, wrapping
        back to the start after the last -- instead of sampling shock_types/
        magnitude_range/resource_removal_magnitude_range at random (those are
        then ignored). Duration is still drawn from duration_range. The
        position in the cycle is NOT stored here: this config object is shared
        by the training and eval envs, so each gym_env keeps its own cursor
        and passes it in as cycle_index.

    Eval mode parameters
    --------------------
    eval_onset_time  : float   Fixed sim-time (seconds) when the shock fires.
    eval_shock_type  : str     'RESOURCE_REMOVAL' or 'DEMAND_SPIKE'.
    eval_magnitude   : float   Shock magnitude (fraction).
    eval_duration    : float   Shock duration (seconds).

    Common
    ------
    mode             : str     'train' or 'eval'.
    resource_groups  : dict    Maps group name → list of resource names.
                               Used by RESOURCE_REMOVAL to know which
                               resources belong to each group. Takes priority
                               over resource_removal_scope if non-empty.
    resource_removal_scope : str
        'activity' (default) or 'global'. Empirically, 'global' (uniform-
        random draw from the whole registered resource pool) produced no
        measurable disruption even at 6h duration on the single most
        load-bearing resource — with a small pool (~19-22 resources) and load
        concentrated on 2-3 of them, a global draw usually misses what
        matters, and even a direct hit is diluted across a system with slack
        everywhere else. 'activity' instead removes a fraction of the
        resources eligible for one specific activity (that activity's own
        pool is typically only ~6 people, from the tier-based pool design),
        so the same magnitude is a much bigger fractional hit where it can
        actually create a queue.
    eval_target_activity : str, optional
        Eval mode + scope='activity' only: which activity to target. If
        None, one is chosen at random from the dataset's real (non-
        placeholder-only) activities.
    """
    mode: str = 'train'

    # train
    lambda_rate: float = 1 / 3600.0   # one shock per hour on average
    shock_types: list = field(default_factory=lambda: ['RESOURCE_REMOVAL', 'DEMAND_SPIKE'])
    magnitude_range: tuple = (0.2, 0.5)   # DEMAND_SPIKE only -- see docstring
    resource_removal_magnitude_range: tuple = (0.2, 0.5)   # RESOURCE_REMOVAL only -- see docstring
    duration_range: tuple = (1800.0, 7200.0)
    cycle_schedule: list = field(default_factory=list)   # ordered (type, magnitude) cycle -- see docstring

    # eval
    eval_onset_time: float = 3600.0
    eval_shock_type: str = 'RESOURCE_REMOVAL'
    eval_magnitude: float = 0.3
    eval_duration: float = 3600.0
    eval_target_activity: Optional[str] = None

    # shared
    resource_groups: dict = field(default_factory=dict)
    resource_removal_scope: str = 'activity'

    def __post_init__(self):
        # Same failure mode resource_removal_magnitude_range exists to prevent:
        # a RESOURCE_REMOVAL magnitude outside [0, 1] silently clamps to
        # removing an activity's whole eligible pool.
        for shock_type, magnitude in self.cycle_schedule:
            if shock_type not in ('RESOURCE_REMOVAL', 'DEMAND_SPIKE'):
                raise ValueError(f"cycle_schedule: unknown shock type '{shock_type}'")
            if shock_type == 'RESOURCE_REMOVAL' and not 0.0 <= magnitude <= 1.0:
                raise ValueError(
                    f"cycle_schedule: RESOURCE_REMOVAL magnitude {magnitude} is a fraction "
                    f"of the eligible pool and must lie in [0, 1]"
                )


def shock_key(shock_type: str, magnitude: float) -> str:
    """Canonical key for one shock configuration e_j, e.g. 'DEMAND_SPIKE:2' or
    'RESOURCE_REMOVAL:0.2' -- shared by calibrate_resilience_constants.py (which
    writes the per-shock CPL_base table) and gym_env.py (which looks it up)."""
    return f"{shock_type}:{float(magnitude):g}"


# ---------------------------------------------------------------------------
# Runtime state (mutated throughout an episode)
# ---------------------------------------------------------------------------

class ShockState:
    """
    Mutable live state of the shock layer. Read by gym_env at every step to
    build the observation and enforce action masking.
    """

    def __init__(self):
        self.is_active: bool = False
        self.active_type: Optional[str] = None
        self.active_magnitude: float = 0.0
        self.removed_resources: set = set()   # populated by RESOURCE_REMOVAL
        self.removal_target_activity: Optional[str] = None  # set when scope='activity'

    def activate(self, shock_type: str, magnitude: float):
        self.is_active = True
        self.active_type = shock_type
        self.active_magnitude = magnitude

    def deactivate(self):
        self.is_active = False
        self.active_type = None
        self.active_magnitude = 0.0
        self.removed_resources.clear()
        self.removal_target_activity = None

    def __repr__(self):
        return (f"ShockState(active={self.is_active}, type={self.active_type}, "
                f"magnitude={self.active_magnitude:.2f}, "
                f"removed={self.removed_resources})")


# ---------------------------------------------------------------------------
# Scheduler (simpy process)
# ---------------------------------------------------------------------------

def shock_scheduler(env, config: ShockConfig, state: ShockState, simulation_process):
    """
    SimPy generator process.  Registered once per episode in gym_env.reset().

    Train mode: fires shocks at Poisson-distributed intervals until the
    episode ends (simpy terminates the generator when the env finishes).
    Multiple shocks can overlap — each onset/offset pair is independent.

    Eval mode: fires exactly one shock at config.eval_onset_time, waits for
    its duration, then deactivates and exits.
    """
    if config.mode == 'eval':
        yield env.timeout(config.eval_onset_time)
        _apply_onset(env, config.eval_shock_type, config.eval_magnitude,
                     config, state, simulation_process)
        yield env.timeout(config.eval_duration)
        _apply_offset(config.eval_shock_type, state, simulation_process)

    else:  # train
        while True:
            # Inter-shock interval ~ Exponential(lambda_rate)
            interval = random.expovariate(config.lambda_rate)
            yield env.timeout(interval)

            shock_type = random.choice(config.shock_types)
            magnitude_range = (
                config.resource_removal_magnitude_range if shock_type == 'RESOURCE_REMOVAL'
                else config.magnitude_range
            )
            magnitude = random.uniform(*magnitude_range)
            duration = random.uniform(*config.duration_range)

            _apply_onset(env, shock_type, magnitude, config, state, simulation_process)
            yield env.timeout(duration)
            _apply_offset(shock_type, state, simulation_process)


def fire_single_shock(env, config: ShockConfig, state: ShockState, simulation_process,
                      cycle_index: Optional[int] = None):
    """
    Fire exactly one shock, immediately (no inter-shock interval draw, no
    recurrence) — used by gym_env.reset() for the single-shock-per-episode
    training structure. This replaces shock_scheduler for training: that
    function's infinite Poisson-recurring loop in 'train' mode is exactly what
    made next_decision_moment()'s search unbounded (env.peek() could never
    return math.inf, since another shock was always scheduled). Firing once
    and never re-arming removes that failure mode entirely.

    In 'eval' mode, uses the fixed eval_shock_type/eval_magnitude/eval_duration
    fields (reproducible, controlled shock) rather than random sampling — the
    onset *timing* is still decided by the caller (gym_env's warm-up
    criterion), only type/magnitude/duration are fixed. In 'train' mode,
    takes entry cycle_index (mod its length) of config.cycle_schedule if one
    is set, otherwise samples the same way shock_scheduler's train branch
    already does.

    Returns (shock_type, magnitude, duration) for logging/diagnostics.
    """
    if config.mode == 'eval':
        shock_type = config.eval_shock_type
        magnitude = config.eval_magnitude
        duration = config.eval_duration
    elif config.cycle_schedule:
        if cycle_index is None:
            raise ValueError("config.cycle_schedule is set but no cycle_index was given")
        shock_type, magnitude = config.cycle_schedule[cycle_index % len(config.cycle_schedule)]
        duration = random.uniform(*config.duration_range)
    else:
        shock_type = random.choice(config.shock_types)
        magnitude_range = (
            config.resource_removal_magnitude_range if shock_type == 'RESOURCE_REMOVAL'
            else config.magnitude_range
        )
        magnitude = random.uniform(*magnitude_range)
        duration = random.uniform(*config.duration_range)

    _apply_onset(env, shock_type, magnitude, config, state, simulation_process)
    env.process(_delayed_single_offset(env, duration, shock_type, state, simulation_process))
    return shock_type, magnitude, duration


def _delayed_single_offset(env, duration, shock_type, state, simulation_process):
    """One-shot SimPy process: waits `duration`, then applies the offset. No loop."""
    yield env.timeout(duration)
    _apply_offset(shock_type, state, simulation_process)


# ---------------------------------------------------------------------------
# Onset / offset helpers
# ---------------------------------------------------------------------------

def _apply_onset(env, shock_type: str, magnitude: float,
                 config: ShockConfig, state: ShockState, simulation_process):
    """Apply the shock effect and update ShockState."""
    state.activate(shock_type, magnitude)

    if shock_type == 'RESOURCE_REMOVAL':
        # Pick the candidate pool to remove from, in priority order:
        # 1) an explicit resource_groups entry, if configured;
        # 2) an activity's own eligible-resource pool (scope='activity',
        #    the default — see ShockConfig docstring for why: a global draw
        #    on a small pool with concentrated load usually misses what
        #    matters);
        # 3) the whole registered resource pool (scope='global', legacy
        #    behaviour, kept for comparison).
        if config.resource_groups:
            group_name = random.choice(list(config.resource_groups.keys()))
            pool = config.resource_groups[group_name]
        elif config.resource_removal_scope == 'activity':
            pt_map = simulation_process.resource_processing_times
            # Activities with at least one real (non-placeholder) eligible
            # resource -- placeholder-only activities have nothing to remove.
            real_activities = sorted({
                act for res, acts in pt_map.items() if res != PLACEHOLDER_RESOURCE
                for act in acts
            })
            if config.mode == 'eval' and config.eval_target_activity:
                target_activity = config.eval_target_activity
            elif real_activities:
                target_activity = random.choice(real_activities)
            else:
                target_activity = None
            state.removal_target_activity = target_activity
            if target_activity is not None:
                pool = [
                    r for r, acts in pt_map.items()
                    if r != PLACEHOLDER_RESOURCE and target_activity in acts
                ]
            else:
                pool = []
        else:
            pool = list(simulation_process._resources.keys())
            # PLACEHOLDER_RESOURCE must always stay available — it's the
            # deadlock-avoidance fallback for zero-eligible-resource activities
            # (Module 1 activities_at_risk), not a real resource to be shocked.
            pool = [r for r in pool if r not in ('TRIGGER_TIMER', PLACEHOLDER_RESOURCE)]

        if pool:
            n_remove = max(1, math.floor(len(pool) * magnitude))
            # Choose which resources to remove (not already removed).
            available = [r for r in pool if r not in state.removed_resources]
            to_remove = random.sample(available, min(n_remove, len(available)))
            for res_name in to_remove:
                state.removed_resources.add(res_name)
                role_sim = simulation_process._resources.get(res_name)
                if role_sim is not None:
                    role_sim.mark_removed()

    elif shock_type == 'DEMAND_SPIKE':
        simulation_process.arrival_multiplier = 1.0 + magnitude
        # D-demand-spike-fix: all arrivals are scheduled upfront (before this
        # shock existed), each waiting out a fixed duration computed under
        # multiplier=1.0 — changing arrival_multiplier alone does nothing to
        # a wait already in progress. Interrupt every not-yet-arrived token's
        # wait so it recomputes its remaining time under the new multiplier
        # (Token.inter_trigger_time, event_trace.py).
        _interrupt_pending_arrivals(simulation_process)


def _apply_offset(shock_type: str, state: ShockState, simulation_process):
    """Undo the shock effect and clear ShockState."""
    if shock_type == 'RESOURCE_REMOVAL':
        for res_name in list(state.removed_resources):
            role_sim = simulation_process._resources.get(res_name)
            if role_sim is not None:
                role_sim.mark_restored()

    elif shock_type == 'DEMAND_SPIKE':
        simulation_process.arrival_multiplier = 1.0
        # Symmetric with onset: still-waiting tokens must also decompress
        # back to normal pace for whatever time remains after the shock ends.
        _interrupt_pending_arrivals(simulation_process)

    state.deactivate()


def _interrupt_pending_arrivals(simulation_process):
    """Wake every not-yet-arrived token's inter_trigger_time() wait so it
    re-reads arrival_multiplier and recomputes its remaining delay. Silently
    skips any process that has already finished (token already arrived) —
    pending_arrival_procs may lag by exactly one event-loop tick behind
    Token.inter_trigger_time's own cleanup, since interrupting is not
    synchronous with the token popping itself out of the dict."""
    for token_id, proc in list(simulation_process.pending_arrival_procs.items()):
        try:
            proc.interrupt()
        except RuntimeError:
            simulation_process.pending_arrival_procs.pop(token_id, None)
