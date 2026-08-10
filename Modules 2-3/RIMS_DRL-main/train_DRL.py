from collections import deque
from subprocess import call
import gymnasium as gym
import os
import numpy as np
from gym_env import gym_env
import datetime
import sys
import json

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy, MaskableMultiInputActorCriticPolicy
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from callbacks import SaveOnBestTrainingRewardCallback, EvalPolicyCallback
from callbacks import custom_schedule, linear_schedule


class CustomPolicy(MaskableActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=dict(pi=[256, 256],
                                                          vf=[256, 256]))

if len(sys.argv) > 1:
    NAME_LOG = sys.argv[1]
    if not sys.argv[2] == 'from_input_data':
        N_TRACES = int(sys.argv[2])
    else:
        N_TRACES = sys.argv[2]
    CALENDAR = True if sys.argv[3] == "True" else False
    threshold = int(sys.argv[4])
    # sys.argv[5] is parsed for CLI positional-arg compatibility but ignored below:
    # the Action Space spec (Module_3 doc §2) has no postpone/no-op action, so every
    # gym_env(...) construction site below hardcodes postpone=False.
    _ = sys.argv[5]
    reward_function = sys.argv[6]
    artifact_path = sys.argv[7] if len(sys.argv) > 7 else None
    # pool_override_path/calibration_path: the curated resource pool and the
    # fixed P*/cpl_base/phi_fix calibration file this run must use (see
    # calibrate_resilience_constants.py). Both required whenever shock_config
    # is not None -- see the loud failure below, not a silent recompute.
    pool_override_path = sys.argv[8] if len(sys.argv) > 8 else None
    calibration_path = sys.argv[9] if len(sys.argv) > 9 else None
else:
    NAME_LOG = 'BPI_Challenge_2017_W_Two_TS'
    N_TRACES = 'from_input_data'
    CALENDAR = False   # D10: uniform calendars — always False for resilience experiments
    threshold = 20
    reward_function = 'RF2'
    artifact_path = None   # set this to the Module 1 artifact path before running
    pool_override_path = None
    calibration_path = None

# D9/D6: RF2's CPL budget fraction (CPL_max = p * CPL_base) and RF1's r1/r2 combination
# weight. Defaults per Module_3_DRL_Environment_and_Agent.md §5/§8. Must match
# whatever calibrate_resilience_constants.py used for cpl_base -> cpl_budget.
p_cpl_fraction = 0.5
beta = 0.5


# ---------------------------------------------------------------------------
# Shock configuration (D5) — set to None to disable shocks entirely.
# Import ShockConfig here and customise the fields before running.
#
# D-single-shock: gym_env now fires exactly one shock per episode itself (right
# after its hidden warm-up phase, via shock.fire_single_shock), instead of using
# this config's old recurring Poisson-process scheduler. type/magnitude/duration
# are still sampled from shock_types/magnitude_range/duration_range below;
# `lambda_rate` is no longer read anywhere in this flow (it only mattered for
# the old recurring scheduler's inter-shock interval) — left here for reference
# only, in case some other script still uses the recurring scheduler directly.
# ---------------------------------------------------------------------------
from shock import ShockConfig
shock_config = ShockConfig(
    mode='train',
    # RESOURCE_REMOVAL re-enabled: it previously shared magnitude_range with
    # DEMAND_SPIKE below, but for RESOURCE_REMOVAL magnitude means "fraction of
    # the pool to remove" (not a multiplier) -- drawing from DEMAND_SPIKE's 5-8
    # range clamped every RESOURCE_REMOVAL shock to removing 100% of an
    # activity's eligible resources, which could strand a pending token with
    # zero eligible resources for the rest of the episode. Found via the
    # Consulta smoke test (IndexError in gym_env.step() shortly after a
    # RESOURCE_REMOVAL shock). Fixed at the root in shock.py: ShockConfig now
    # has a separate resource_removal_magnitude_range field, used only for
    # RESOURCE_REMOVAL draws; magnitude_range below is DEMAND_SPIKE-only.
    shock_types=['RESOURCE_REMOVAL', 'DEMAND_SPIKE'],
    # magnitude_range (DEMAND_SPIKE only): was (0.2, 0.5) -- a 1.2x-1.5x arrival
    # bump, far milder than what calibration/eval actually use. Every P*/
    # cpl_base/phi_fix calibration and resilience-metric run this project has
    # produced (calibrate_resilience_constants.py, compute_resilience_metrics.py)
    # uses magnitude in [5, 8] (a 6x-9x arrival multiplier -- DEMAND_SPIKE's
    # multiplier is 1+magnitude), the validated realistic range
    # (Training_Readiness_Implementation_Summary.md SS32). Training on (0.2, 0.5)
    # while being evaluated against [5, 8] would mean the agent rarely, if ever,
    # practices on a shock severe enough to need a real recovery response during
    # training. Matched to [5, 8] here so training and eval see the same
    # disruption regime.
    magnitude_range=(5.0, 8.0),
    # resource_removal_magnitude_range (RESOURCE_REMOVAL only): fraction of an
    # activity's eligible resource pool to remove, must stay within [0, 1].
    # (0.2, 0.5) is this project's original, unchanged RESOURCE_REMOVAL range --
    # unlike DEMAND_SPIKE, RESOURCE_REMOVAL was never recalibrated against a
    # different eval magnitude, so there's no "match eval" argument for moving
    # this one.
    resource_removal_magnitude_range=(0.2, 0.5),
    # duration_range: was (1800.0, 7200.0) (30min-2h) -- calibration/eval both use
    # a fixed 21,600s (6h) shock (calibrate_resilience_constants.py,
    # compute_resilience_metrics.py). Fixed here to the same 21,600s rather than
    # sampled, for the same reason as magnitude_range above: training should see
    # the same disruption regime it's evaluated against.
    duration_range=(21600.0, 21600.0),
    lambda_rate=1 / 7200.0,   # unused by the single-shock flow; see note above
)
# Set to None to train without shocks (baseline comparison).
# shock_config = None


if __name__ == '__main__':
    load_model = False
    postpone_penalty = 0
    time_steps = 10240000
    n_steps = {
        "BPI_Challenge_2012_W_Two_TS": 10240,
        "confidential_1000":           5120,
        "ConsultaDataMining201618":    5120,
        "PurchasingExample":           5120,
        "BPI_Challenge_2017_W_Two_TS": 48128,
        "Productions":                 5120,
    }
    n_steps = n_steps.get(NAME_LOG, 5120)  # fallback for datasets not in table

    # Create log dir — include shock flag so runs are distinguishable (D5).
    shock_tag = 'S1' if shock_config is not None else 'S0'
    log_dir = f"tmp_training_2/{NAME_LOG}_{N_TRACES}_C{CALENDAR}_T{threshold}_PFalse_{reward_function}_{shock_tag}/"
    os.makedirs(log_dir, exist_ok=True)

    # ---------------------------------------------------------------------------
    # Fixed calibration constants (P*, cpl_base -> cpl_budget, phi_fix): loaded
    # from a file produced once by calibrate_resilience_constants.py, never
    # recomputed live here. A live recompute on every launch was found to risk
    # inconsistency across runs (different pool, different random draw, etc.);
    # the whole point of this file is that every simulation on a given
    # (dataset, pool) pair — calibration, training, eval — uses the exact same
    # numbers. Fails loudly rather than silently falling back to zeros/None if
    # shocks are enabled and no calibration file was given, or if the file's
    # own recorded dataset/pool don't match what was requested here.
    # ---------------------------------------------------------------------------
    p_star_override = None
    cpl_budget = 0.0
    phi_fix = None
    if shock_config is not None:
        if calibration_path is None:
            raise ValueError(
                "shock_config is set but no calibration_path was given. Run "
                "calibrate_resilience_constants.py for this (dataset, pool) pair first, "
                "then pass its output path as the 9th CLI argument."
            )
        with open(calibration_path, 'r') as f:
            calibration = json.load(f)
        cfg = calibration.get('config_used', {})
        if cfg.get('dataset') != NAME_LOG:
            raise ValueError(
                f"Calibration file {calibration_path} was computed for dataset "
                f"'{cfg.get('dataset')}', not '{NAME_LOG}'. Refusing to reuse a "
                f"mismatched calibration."
            )
        if pool_override_path is not None and cfg.get('pool') != pool_override_path:
            raise ValueError(
                f"Calibration file {calibration_path} was computed for pool "
                f"'{cfg.get('pool')}', not '{pool_override_path}'. Refusing to reuse a "
                f"mismatched calibration."
            )
        p_star_override = calibration['P_star']
        cpl_budget = p_cpl_fraction * calibration['cpl_base']   # CPL_max = p * CPL_base
        phi_fix = calibration['phi_fix']
        print(f"Loaded calibration from {calibration_path}: "
              f"P_star={p_star_override:.4f} cpl_base={calibration['cpl_base']:.4f} "
              f"cpl_budget(p={p_cpl_fraction})={cpl_budget:.4f} phi_fix={phi_fix:.4f}", flush=True)

    # Create and wrap the environment. pool_override_path/p_star_override keep
    # training on the exact same curated pool and fixed P* the calibration file
    # above was computed from (both None -- i.e. no-op -- if shocks are disabled).
    env_simulator = gym_env(
        NAME_LOG, N_TRACES, CALENDAR,
        threshold=threshold, postpone=False,
        reward_function=reward_function,
        artifact_path=artifact_path,
        shock_config=shock_config,
        cpl_budget=cpl_budget,
        eta_lambda=0.01,
        phi_fix=phi_fix,
        beta=beta,
        pool_override_path=pool_override_path,
        p_star_override=p_star_override,
    )
    env = Monitor(env_simulator, log_dir)

    # Create the model.
    gamma = 0.999 if reward_function in ('inverse_CT', 'RF2') else 1
    model = MaskablePPO(CustomPolicy, env_simulator, clip_range=0.2, learning_rate=linear_schedule(3e-4), n_steps=int(n_steps), batch_size=256, gamma=gamma, verbose=1)

    # Logging to tensorboard.
    # tensorboard --logdir ./tmp_training_2/
    model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

    # Eval env shares the same shock_config, pool, and P* so evaluation episodes
    # include shocks and are directly comparable to training episodes.
    eval_env = gym_env(
        NAME_LOG, N_TRACES, CALENDAR,
        threshold=threshold, postpone=False,
        reward_function=reward_function,
        artifact_path=artifact_path,
        shock_config=shock_config,
        cpl_budget=cpl_budget,
        eta_lambda=0.01,
        phi_fix=phi_fix,
        beta=beta,
        pool_override_path=pool_override_path,
        p_star_override=p_star_override,
    )
    eval_env = Monitor(eval_env, log_dir)
    nr_evaluations = 10 if NAME_LOG != 'BPI_Challenge_2017_W_Two_TS' else 3
    eval_callback = EvalPolicyCallback(check_freq=5*int(n_steps), nr_evaluations=nr_evaluations, log_dir=log_dir, eval_env=eval_env)

    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    model.learn(total_timesteps=int(time_steps), callback=eval_callback)

    model.save(f'{log_dir}/model_final')
