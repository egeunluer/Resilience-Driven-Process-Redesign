from collections import deque
from subprocess import call
import gymnasium as gym
import os
import numpy as np
from gym_env import gym_env
import datetime
import sys

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy, MaskableMultiInputActorCriticPolicy
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure

from gymnasium.wrappers import normalize
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from callbacks import SaveOnBestTrainingRewardCallback, EvalPolicyCallback
from callbacks import custom_schedule, linear_schedule


class CustomPolicy(MaskableActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[256, 256],
                                                          vf=[256, 256])])

if len(sys.argv) > 1:
    NAME_LOG = sys.argv[1]
    if not sys.argv[2] == 'from_input_data':
        N_TRACES = int(sys.argv[2])
    else:
        N_TRACES = sys.argv[2]
    CALENDAR = True if sys.argv[3] == "True" else False
    threshold = int(sys.argv[4])
    postpone = True if sys.argv[5] == "True" else False
    reward_function = sys.argv[6]
    artifact_path = sys.argv[7] if len(sys.argv) > 7 else None
else:
    NAME_LOG = 'BPI_Challenge_2017_W_Two_TS'
    N_TRACES = 'from_input_data'
    CALENDAR = False   # D10: uniform calendars — always False for resilience experiments
    threshold = 20
    postpone = True
    reward_function = 'RF2'
    artifact_path = None   # set this to the Module 1 artifact path before running


# ---------------------------------------------------------------------------
# Shock configuration (D5) — set to None to disable shocks entirely.
# Import ShockConfig here and customise the fields before running.
# ---------------------------------------------------------------------------
from shock import ShockConfig
shock_config = ShockConfig(
    mode='train',
    shock_types=['RESOURCE_REMOVAL', 'DEMAND_SPIKE'],
    magnitude_range=(0.2, 0.5),
    duration_range=(1800.0, 7200.0),
    lambda_rate=1 / 7200.0,   # one shock per 2 hours on average
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
    log_dir = f"tmp_training_2/{NAME_LOG}_{N_TRACES}_C{CALENDAR}_T{threshold}_P{postpone}_{reward_function}_{shock_tag}/"
    os.makedirs(log_dir, exist_ok=True)

    # ---------------------------------------------------------------------------
    # FIFO CPL calibration (D6): run one episode with a first-valid-action policy
    # to establish a baseline cpl_budget for the RF2 Lagrange constraint.
    # ---------------------------------------------------------------------------
    cpl_budget = 0.0
    if reward_function == 'RF2' and shock_config is not None:
        print('Running FIFO calibration episode for CPL baseline...', flush=True)
        cal_env = gym_env(
            NAME_LOG, N_TRACES, CALENDAR,
            threshold=threshold, postpone=False,
            reward_function='inverse_CT',
            artifact_path=artifact_path,
            shock_config=shock_config,
        )
        obs, _ = cal_env.reset()
        done = False
        while not done:
            mask = cal_env.action_masks()
            valid = [i for i, m in enumerate(mask) if m]
            action = valid[0] if valid else 0   # always pick first eligible action (FIFO)
            obs, _r, done, _, _ = cal_env.step(action)
        cpl_budget = cal_env.cpl_accumulated
        print(f'FIFO CPL baseline (cpl_budget): {cpl_budget:.4f}', flush=True)

    # Create and wrap the environment.
    env_simulator = gym_env(
        NAME_LOG, N_TRACES, CALENDAR,
        threshold=threshold, postpone=postpone,
        reward_function=reward_function,
        artifact_path=artifact_path,
        shock_config=shock_config,
        cpl_budget=cpl_budget,
        eta_lambda=0.01,
    )
    env = Monitor(env_simulator, log_dir)

    # Create the model.
    gamma = 0.999 if reward_function in ('inverse_CT', 'RF2') else 1
    model = MaskablePPO(CustomPolicy, env_simulator, clip_range=0.2, learning_rate=linear_schedule(3e-4), n_steps=int(n_steps), batch_size=256, gamma=gamma, verbose=1)

    # Logging to tensorboard.
    # tensorboard --logdir ./tmp_training_2/
    model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

    # Eval env shares the same shock_config so evaluation episodes include shocks.
    eval_env = gym_env(
        NAME_LOG, N_TRACES, CALENDAR,
        threshold=threshold, postpone=postpone,
        reward_function=reward_function,
        artifact_path=artifact_path,
        shock_config=shock_config,
        cpl_budget=cpl_budget,
        eta_lambda=0.01,
    )
    eval_env = Monitor(eval_env, log_dir)
    nr_evaluations = 10 if NAME_LOG != 'BPI_Challenge_2017_W_Two_TS' else 3
    eval_callback = EvalPolicyCallback(check_freq=5*int(n_steps), nr_evaluations=nr_evaluations, log_dir=log_dir, eval_env=eval_env)

    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    model.learn(total_timesteps=int(time_steps), callback=eval_callback)

    model.save(f'{log_dir}/model_final')
