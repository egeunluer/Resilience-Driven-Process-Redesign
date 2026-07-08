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
        (min, max) fraction for both shock types.  For RESOURCE_REMOVAL this
        is the fraction of resources removed; for DEMAND_SPIKE it is the
        fractional rate increase (so 0.5 → arrival rate × 1.5).
    duration_range  : tuple[float, float]
        (min, max) shock duration in simulation seconds.

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
                               resources belong to each group.
    """
    mode: str = 'train'

    # train
    lambda_rate: float = 1 / 3600.0   # one shock per hour on average
    shock_types: list = field(default_factory=lambda: ['RESOURCE_REMOVAL', 'DEMAND_SPIKE'])
    magnitude_range: tuple = (0.2, 0.5)
    duration_range: tuple = (1800.0, 7200.0)

    # eval
    eval_onset_time: float = 3600.0
    eval_shock_type: str = 'RESOURCE_REMOVAL'
    eval_magnitude: float = 0.3
    eval_duration: float = 3600.0

    # shared
    resource_groups: dict = field(default_factory=dict)


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

    def activate(self, shock_type: str, magnitude: float):
        self.is_active = True
        self.active_type = shock_type
        self.active_magnitude = magnitude

    def deactivate(self):
        self.is_active = False
        self.active_type = None
        self.active_magnitude = 0.0
        self.removed_resources.clear()

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
            magnitude = random.uniform(*config.magnitude_range)
            duration = random.uniform(*config.duration_range)

            _apply_onset(env, shock_type, magnitude, config, state, simulation_process)
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
        # Pick a random resource group (or all resources if none defined).
        if config.resource_groups:
            group_name = random.choice(list(config.resource_groups.keys()))
            pool = config.resource_groups[group_name]
        else:
            pool = list(simulation_process._resources.keys())
            pool = [r for r in pool if r != 'TRIGGER_TIMER']

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
        # InterTriggerTimer reads arrival_multiplier on each draw.
        simulation_process.arrival_multiplier = 1.0 + magnitude


def _apply_offset(shock_type: str, state: ShockState, simulation_process):
    """Undo the shock effect and clear ShockState."""
    if shock_type == 'RESOURCE_REMOVAL':
        for res_name in list(state.removed_resources):
            role_sim = simulation_process._resources.get(res_name)
            if role_sim is not None:
                role_sim.mark_restored()

    elif shock_type == 'DEMAND_SPIKE':
        simulation_process.arrival_multiplier = 1.0

    state.deactivate()
