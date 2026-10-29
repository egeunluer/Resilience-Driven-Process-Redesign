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

from callbacks import SaveOnBestTrainingRewardCallback, EvalPolicyCallback, ResilienceCallback
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
    # Wired for Consulta's CALENDAR=True min3activities_plus157 pool
    # (RIMS_DRL_Modification_Plan.md Step 60-64): CALENDAR=True alone (no
    # shock) roughly quadruples P* (25,946s -> 109,604.5s) -- two of
    # Consulta's 11 real activities have only 1-2 eligible resources, both
    # defaulting to a synthetic Mon-Fri 08:00-18:00 window with no mined
    # calendar of their own. D10's original rationale (don't let shift
    # patterns confound the shock signal) still holds for BPI2017/BPI12W/
    # other pools that haven't had this same investigation -- this branch is
    # Consulta-specific, not a dataset-agnostic default.
    NAME_LOG = 'ConsultaDataMining201618'
    # N_TRACES: NOT 'from_input_data' (181, Consulta's real case count).
    # Training_Readiness_Implementation_Summary.md SS30 found a strong,
    # long demand spike can exhaust the whole 181-case population before the
    # backlog drains, which reads as "never recovered" for a reason
    # unrelated to resilience -- not a hard ceiling, since Consulta's
    # 'custom' (Prophet) arrival generator extrapolates arbitrarily far
    # (SS30: N_TRACES=400 fixed this with no code changes). Set to 2000 to
    # match calibrate_resilience_constants.py's own n_traces for this pool
    # (consulta_min3activities_plus157_calendar_calibration.json).
    N_TRACES = 2000
    CALENDAR = True
    # threshold=20 matches the calibration file's config_used.threshold
    # exactly; functionally inert here since pool_override_path below
    # replaces self.resources regardless of which input_<name>.json loads.
    threshold = 20
    # RF1: user's standing default reward function when unspecified
    # (Training_Readiness_Implementation_Summary.md SS48), overriding
    # train_DRL.py's own historical no-argv default of RF2.
    reward_function = 'RF2'
    artifact_path = '../../Module 1/artifacts/consulta.json'
    pool_override_path = '../../Module 1/artifacts/consulta_min3activities_plus157_pool.json'
    calibration_path = '../../Module 1/artifacts/consulta_min3activities_plus157_calendar_calibration.json'

# RF2 (revised, RF2_reward_function.md): r_k = (-D_k + rf2_epsilon * H_k) / CPL_base[e_j],
# headroom H uncapped. p_cpl_fraction is evaluation-only (CPL_max = p * CPL_base[e_j],
# u = B_K / CPL_max logged at episode end) -- never part of the reward.
rf2_epsilon = 0.5
p_cpl_fraction = 0.5
# RF1's r1/r2 combination weight (Module_3_DRL_Environment_and_Agent.md §8).
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
from shock import ShockConfig, shock_key
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
    # bump, far milder than what calibration/eval actually use. The original
    # calendar=False calibration across this project used magnitude in [5, 8]
    # (a 6x-9x arrival multiplier -- DEMAND_SPIKE's multiplier is
    # 1+magnitude). For Consulta's CALENDAR=True min3activities_plus157 pool
    # specifically, a compute_resilience_metrics.py scan at magnitudes 2/3/4
    # (duration=360,000s/100h, 2 seeds each) found every one of those milder
    # magnitudes already breaches P* and recovers -- this calendar-on pool is
    # fragile at 3x-5x arrivals, not just 6x-9x. Narrowed to [2, 5] so the
    # agent trains across the magnitudes actually shown to matter for this
    # pool, rather than only the far end of the range. Note: cpl_base/phi_fix
    # in the calibration file below were still computed at a single fixed
    # magnitude=6 point (calibrate_resilience_constants.py never resampled
    # across a range) -- those anchors remain valid as fixed reference values
    # regardless of what range training itself samples from.
    magnitude_range=(2.0, 5.0),
    # resource_removal_magnitude_range (RESOURCE_REMOVAL only): fraction of an
    # activity's eligible resource pool to remove, must stay within [0, 1].
    # (0.2, 0.5) is this project's original, unchanged RESOURCE_REMOVAL range --
    # unlike DEMAND_SPIKE, RESOURCE_REMOVAL was never recalibrated against a
    # different eval magnitude, so there's no "match eval" argument for moving
    # this one.
    resource_removal_magnitude_range=(0.2, 0.5),
    # resource_removal_scope: 'global' -- magnitude is the fraction of the
    # WHOLE curated pool removed (e.g. 0.5 of 22 resources -> 11), not of one
    # randomly chosen activity's eligible resources ('activity', the
    # ShockConfig default). Activity scope was found to remove only ~1.6-3.7
    # resources on average for Consulta (5 of 11 activities have <=3 eligible
    # resources, so magnitude barely changed anything) and produced no
    # measurable phi(t) disruption at any magnitude in [0.2, 0.5].
    resource_removal_scope='global',
    # duration_range: dataset/pool-dependent -- must equal whatever fixed
    # shock_duration calibrate_resilience_constants.py used for the
    # calibration file this run loads (both scripts already enforce
    # dataset/pool/calendar matching a calibration file; duration is not yet
    # cross-checked the same way, so keeping this in sync is on the caller).
    # 21,600s (6h) was correct for every calendar=False calibration on this
    # project (BPI2017, BPI12W, Consulta, etc.) -- unchanged for those.
    # For Consulta with CALENDAR=True specifically, 21,600s was found to be
    # far too short to be a real disruption: with the calendar's baseline
    # congestion alone quadrupling P* (see the CALENDAR comment above), a 6h
    # DEMAND_SPIKE never breaches it at any tested magnitude (5-8x arrivals,
    # 0/8 magnitude x seed combinations breached). A duration scan at fixed
    # magnitude found the breach crossover sits between 2x and 5x the
    # original 20h (=72,000s) baseline duration, i.e. somewhere around
    # 144,000-360,000s -- real cases in this exact log run up to 135.84 days
    # (Camargo et al. 2022, Table 1: Consulta = "ACR", avg 14.89d / max
    # 135.84d case duration), so testing at this scale is not unrealistic.
    # Set to 360,000s (100h), comfortably past the crossover and matched to
    # the recalibrated consulta_min3activities_plus157_calendar_calibration.json
    # (see RIMS_DRL_Modification_Plan.md Step 60). BPI2017's own CALENDAR=True
    # investigation (Step 65-68) independently used the same 100h duration
    # for its magnitude scan and calibration probes -- also breach-worthy
    # there (P* +62%, cpl_base ~388x), though BPI2017's own calibration file
    # was never regenerated/wired here (Phase 23 was investigation-only).
    # Revert to 21,600.0 for any dataset/pool whose calibration file was
    # computed with calendar=False.
    duration_range=(360000.0, 360000.0),
    lambda_rate=1 / 7200.0,   # unused by the single-shock flow; see note above
    # cycle_schedule: shocks are presented in a fixed order instead of sampled
    # at random -- one entry per episode, wrapping around: DEMAND_SPIKE
    # magnitude 2 -> 3 -> 4 -> 5, then back to 2. Guarantees every magnitude is
    # visited equally often, rather than relying on random draws to cover the
    # range. While this is non-empty, shock_types/magnitude_range/
    # resource_removal_magnitude_range above are ignored; duration_range still
    # applies (fixed at 100h). Set to [] to go back to random sampling.
    # RESOURCE_REMOVAL is deliberately absent: on this pool, global removal of
    # 20-70% of resources for 100h (and even 1 week) never breached P* beyond
    # the no-shock noise level, so its RF2 CPL_base is 0 and the reward can't be
    # normalized by it -- the min-3 pool's versatile resources absorb it fully.
    cycle_schedule=[('DEMAND_SPIKE', m) for m in (2.0, 3.0, 4.0, 5.0)],
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
    # RIMS_LOG_ROOT: where run folders go (TensorBoard events, progress.csv, models).
    # Defaults to tmp_training_2/; on Colab, point it at Google Drive so logs and
    # checkpoints survive a runtime disconnect. RIMS_RUN_TAG: optional suffix so
    # repeated runs of the same configuration get separate folders (and separate
    # TensorBoard curves) instead of appending into one.
    shock_tag = 'S1' if shock_config is not None else 'S0'
    log_root = os.environ.get('RIMS_LOG_ROOT', 'tmp_training_2')
    run_name = f"{NAME_LOG}_{N_TRACES}_C{CALENDAR}_T{threshold}_PFalse_{reward_function}_{shock_tag}"
    if reward_function == 'RF2':
        run_name += f"_eps{rf2_epsilon:g}"
    if os.environ.get('RIMS_RUN_TAG'):
        run_name += f"_{os.environ['RIMS_RUN_TAG']}"
    log_dir = os.path.join(log_root, run_name) + '/'
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to {log_dir}", flush=True)

    # ---------------------------------------------------------------------------
    # Fixed calibration constants (P*, per-shock CPL_base table, phi_fix): loaded
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
    cpl_base_table = None
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
        # calendar: P_star/cpl_base/phi_fix are baseline/shock statistics computed
        # under one specific resource-availability regime (always-on vs.
        # business-hours calendars) -- reusing a calendar=False calibration while
        # actually running CALENDAR=True (or vice versa) silently judges phi(t)
        # against a threshold from a different regime. Older calibration files
        # predate this field entirely; they were all computed with CALENDAR=False
        # (calibrate_resilience_constants.py's build_env hardcoded it), so default
        # the missing case to False rather than skipping the check.
        if cfg.get('calendar', False) != CALENDAR:
            raise ValueError(
                f"Calibration file {calibration_path} was computed with calendar="
                f"{cfg.get('calendar', False)}, not calendar={CALENDAR}. Refusing to reuse a "
                f"mismatched calibration."
            )
        p_star_override = calibration['P_star']
        phi_fix = calibration['phi_fix']
        if reward_function == 'RF2':
            # One CPL_base per shock configuration e_j, keyed by shock_key(). Every
            # shock this run can fire must have a calibrated, non-zero value, under
            # the same duration and removal scope -- refuse rather than normalize a
            # shock's reward by a value calibrated for a different shock.
            cpl_base_table = calibration.get('cpl_base_table')
            if not cpl_base_table:
                raise ValueError(
                    f"RF2 needs a per-shock cpl_base_table, but {calibration_path} has none "
                    f"(it predates the revised RF2). Re-run calibrate_resilience_constants.py."
                )
            if not shock_config.cycle_schedule:
                raise ValueError("RF2 needs shock_config.cycle_schedule: CPL_base is only "
                                 "calibrated for discrete (type, magnitude) shocks.")
            if (cfg.get('shock_duration'), cfg.get('removal_scope')) != \
                    (shock_config.duration_range[0], shock_config.resource_removal_scope) \
                    or shock_config.duration_range[0] != shock_config.duration_range[1]:
                raise ValueError(
                    f"Calibration shock duration/removal scope ({cfg.get('shock_duration')}, "
                    f"{cfg.get('removal_scope')}) don't match this run's fixed "
                    f"({shock_config.duration_range}, {shock_config.resource_removal_scope})."
                )
            for shock_type, magnitude in shock_config.cycle_schedule:
                key = shock_key(shock_type, magnitude)
                if cpl_base_table.get(key, 0.0) <= 0.0:
                    raise ValueError(
                        f"CPL_base[{key}] is missing or zero in {calibration_path}: that shock "
                        f"never breached P* under the random policy, so RF2's reward can't be "
                        f"normalized by it."
                    )
        # D9 (revised): phi(t)'s tumbling-batch size must match what P*/cpl_base/
        # phi_fix above were calibrated under -- read it from the same file rather
        # than letting this script default independently. See gym_env.py's
        # ppi_window_n comment. Older calibration files (pre-dating the rolling-
        # window -> tumbling-batch redesign) won't have this field; fall back to
        # 20 (the old rolling-window size) with a loud warning, since the loaded
        # P*/cpl_base/phi_fix are stale under the new definition in that case.
        ppi_window_n_override = cfg.get('ppi_window_n')
        if ppi_window_n_override is None:
            ppi_window_n_override = 20
            print(f"WARNING: {calibration_path} has no ppi_window_n recorded (pre-dates the "
                  f"tumbling-batch phi(t) redesign) -- falling back to 20. The loaded P*/"
                  f"cpl_base/phi_fix were calibrated under the OLD rolling-window phi(t); "
                  f"re-run calibrate_resilience_constants.py for a consistent result.",
                  flush=True)
        print(f"Loaded calibration from {calibration_path}: "
              f"P_star={p_star_override:.4f} phi_fix={phi_fix:.4f} "
              f"ppi_window_n={ppi_window_n_override} cpl_base_table={cpl_base_table}", flush=True)
    else:
        ppi_window_n_override = None

    # Create and wrap the environment. pool_override_path/p_star_override keep
    # training on the exact same curated pool and fixed P* the calibration file
    # above was computed from (both None -- i.e. no-op -- if shocks are disabled).
    env_simulator = gym_env(
        NAME_LOG, N_TRACES, CALENDAR,
        threshold=threshold, postpone=False,
        reward_function=reward_function,
        artifact_path=artifact_path,
        shock_config=shock_config,
        cpl_base_table=cpl_base_table,
        rf2_epsilon=rf2_epsilon,
        cpl_max_fraction=p_cpl_fraction,
        phi_fix=phi_fix,
        beta=beta,
        pool_override_path=pool_override_path,
        p_star_override=p_star_override,
        ppi_window_n_override=ppi_window_n_override,
    )
    env = Monitor(env_simulator, log_dir)

    # Create the model.
    # gamma=1 for RF1/RF2 (RF2_reward_function.md SS6: the episode return is then
    # exactly the signed phi(t)-area the reward is built from).
    gamma = 0.999 if reward_function == 'inverse_CT' else 1
    model = MaskablePPO(CustomPolicy, env_simulator, clip_range=0.2, learning_rate=linear_schedule(3e-4), n_steps=int(n_steps), batch_size=256, gamma=gamma, verbose=1)

    # Logging to tensorboard (train/, rollout/, resilience/, eval/ scalars).
    # tensorboard --logdir ./tmp_training_2/   (or $RIMS_LOG_ROOT; on Colab: %tensorboard --logdir ...)
    model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

    # Eval env shares the same shock_config, pool, and P* so evaluation episodes
    # include shocks and are directly comparable to training episodes.
    eval_env = gym_env(
        NAME_LOG, N_TRACES, CALENDAR,
        threshold=threshold, postpone=False,
        reward_function=reward_function,
        artifact_path=artifact_path,
        shock_config=shock_config,
        cpl_base_table=cpl_base_table,
        rf2_epsilon=rf2_epsilon,
        cpl_max_fraction=p_cpl_fraction,
        phi_fix=phi_fix,
        beta=beta,
        pool_override_path=pool_override_path,
        p_star_override=p_star_override,
        ppi_window_n_override=ppi_window_n_override,
    )
    eval_env = Monitor(eval_env, log_dir)
    nr_evaluations = 10 if NAME_LOG != 'BPI_Challenge_2017_W_Two_TS' else 3
    # With a shock cycle, evaluate exactly one full cycle per evaluation round:
    # eval_env's own cursor then wraps back to the same starting point every
    # round, so every round scores the policy on the identical set of shocks
    # and "best model" comparisons across rounds stay like-for-like.
    if shock_config is not None and shock_config.cycle_schedule:
        nr_evaluations = len(shock_config.cycle_schedule)
    eval_callback = EvalPolicyCallback(check_freq=5*int(n_steps), nr_evaluations=nr_evaluations, log_dir=log_dir, eval_env=eval_env)

    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    # ResilienceCallback: per-episode CPL ratio / recovery / deviation to TensorBoard.
    model.learn(total_timesteps=int(time_steps), callback=CallbackList([eval_callback, ResilienceCallback()]))

    model.save(f'{log_dir}/model_final')
