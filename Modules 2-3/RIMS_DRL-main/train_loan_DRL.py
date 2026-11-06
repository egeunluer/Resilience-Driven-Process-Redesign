#!/usr/bin/env python3
"""
Phase 9: train the recovery-action agent (MaskablePPO) on loan_env.LoanEnv.

Reward (RIMS_REWARD): 'case' (default) charges each decision its own case's Loss share
and learns on per-case chains (loan_ppo.CaseMaskablePPO, loan_env docstring); 'global'
is the first setup (process-wide Loss between decisions, run loan_ppo_w1_s0), kept for
the comparison.

Setup (lessons from the proof of concept and the m60 runs):
  - constant learning rate, entropy bonus, gamma = 1 (fixed 50-day horizon); 'global':
    reward normalisation (VecNormalize, rewards only: observations are already in
    [-1, 1]); 'case': rewards already in per-decision units, no VecNormalize;
  - N_ENVS environments in parallel, each starting at a different cell of the training
    cycle, so every rollout covers short and long outages of all scenarios;
  - evaluation every EVAL_EVERY updates on fixed episodes (seeds 1000+, 8 per cell,
    deterministic); best_model by the mean objective (Loss/Loss_base + cost/N);
  - every evaluated model is also kept (checkpoints/), since the final model is chosen
    on the separate benchmark set (baselines_loan.py seeds 200+), not on the
    evaluation round;
  - latest_model.zip (+ vecnormalize.pkl) for RIMS_RESUME=1.

Environment variables:
  RIMS_TIME_STEPS (500000), RIMS_LEARNING_RATE (3e-4), RIMS_ENT_COEF (0.01),
  RIMS_W (1.0), RIMS_REWARD (case), RIMS_BETA (0.5), RIMS_WINDOW_H (0: the whole
  horizon; windowed shared Loss, see loan_env), RIMS_TWIN (1: rewards and evaluation
  against the do-nothing twin, see loan_env), RIMS_SHADOW (1: escalations pay the
  backup workers' shadow price, see loan_env), RIMS_GAE_LAMBDA (0.95),
  RIMS_SEED (0), RIMS_N_ENVS (6), RIMS_RUN_TAG, RIMS_RESUME,
  RIMS_LOG_ROOT (tmp_loan_training)

  python3 train_loan_DRL.py
  tensorboard --logdir tmp_loan_training
"""
import json
import os
from functools import partial

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from callbacks import EvalPolicyCallback, LatestCheckpointCallback, ResilienceCallback
from loan_env import LoanEnv
from loan_ppo import CaseMaskablePPO

ART = '../../Module 1/artifacts/'
MODEL = ART + 'bpic2017_stage_model.json'
DISRUPTIONS = ART + 'bpic2017_disruptions.json'
CALIBRATION = ART + 'bpic2017_pool3_resilience_calibration.json'
N_STEPS = 1024          # per environment and update
EVAL_EVERY = 5          # updates between evaluation rounds
EVAL_PER_CELL = 8      # 4 cells -> 32 evaluation episodes; 3 per cell picked lucky checkpoints


def make_env(cycle, w, start=0, monitor=True, reward='case', beta=0.5, window_h=None, twin=True, shadow=True):
    env = LoanEnv(MODEL, DISRUPTIONS, CALIBRATION, cycle, w=w, reward=reward, beta=beta, window_h=window_h,
                  twin=twin, shadow=shadow)
    env.shock_cycle_index = start
    return Monitor(env) if monitor else env


def main():
    time_steps = int(os.environ.get('RIMS_TIME_STEPS', 500_000))
    lr = float(os.environ.get('RIMS_LEARNING_RATE', 3e-4))
    ent_coef = float(os.environ.get('RIMS_ENT_COEF', 0.01))
    w = float(os.environ.get('RIMS_W', 1.0))
    reward = os.environ.get('RIMS_REWARD', 'case')
    beta = float(os.environ.get('RIMS_BETA', 0.5))
    window_h = float(os.environ.get('RIMS_WINDOW_H', 0)) or None
    twin = os.environ.get('RIMS_TWIN', '1') == '1'
    shadow = os.environ.get('RIMS_SHADOW', '1') == '1'
    gae_lambda = float(os.environ.get('RIMS_GAE_LAMBDA', 0.95))
    seed = int(os.environ.get('RIMS_SEED', 0))
    n_envs = int(os.environ.get('RIMS_N_ENVS', 6))
    resume = os.environ.get('RIMS_RESUME', '0') == '1'
    cycle = [tuple(c) for c in json.load(open(DISRUPTIONS))['training_cycle']['cells']]

    run_name = f'loan_ppo_w{w:g}_s{seed}' + (f'_case_b{beta:g}' + (f'_win{window_h:g}h' if window_h and beta else '') + ('' if twin else '_notwin')
                                             if reward == 'case' else '')
    if os.environ.get('RIMS_RUN_TAG'):
        run_name += f"_{os.environ['RIMS_RUN_TAG']}"
    log_dir = os.path.join(os.environ.get('RIMS_LOG_ROOT', 'tmp_loan_training'), run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f'run {run_name}: {time_steps} steps, reward {reward} (beta {beta}, window {window_h or 'horizon'} h, twin {twin}, shadow {shadow}), lr {lr}, ent {ent_coef}, '
          f'gae_lambda {gae_lambda}, w {w}, {n_envs} envs, '
          f'{len(cycle)} cells -> {log_dir}', flush=True)

    # spread the environments over the cycle
    venv = SubprocVecEnv([partial(make_env, cycle, w, i * len(cycle) // n_envs, reward=reward, beta=beta,
                                  window_h=window_h, twin=twin, shadow=shadow)
                          for i in range(n_envs)])
    venv.seed(seed)
    vn_path = os.path.join(log_dir, 'vecnormalize.pkl')
    latest = os.path.join(log_dir, 'latest_model.zip')
    algo = CaseMaskablePPO if reward == 'case' else MaskablePPO
    if resume:
        if not os.path.exists(latest):
            raise FileNotFoundError(f'RIMS_RESUME=1 but {latest} does not exist')
        if reward == 'global':
            venv = VecNormalize.load(vn_path, venv)
        model = algo.load(latest, env=venv, custom_objects={'learning_rate': lr})
        print(f'resumed from {latest} at {model.num_timesteps} steps', flush=True)
    else:
        if reward == 'global':
            venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=1.0)
        model = algo('MlpPolicy', venv, learning_rate=lr, n_steps=N_STEPS, batch_size=256,
                     gamma=1.0, gae_lambda=gae_lambda, ent_coef=ent_coef, seed=seed, verbose=1)
    model.set_logger(configure(log_dir, ['stdout', 'csv', 'tensorboard']))

    update_calls = N_STEPS          # callback calls per update (one call = n_envs steps)
    eval_env = make_env(cycle, w, monitor=False, reward=reward, beta=beta, window_h=window_h, twin=twin,
                        shadow=shadow)
    n_eval = EVAL_PER_CELL * len(cycle)
    eval_cb = EvalPolicyCallback(check_freq=EVAL_EVERY * update_calls, nr_evaluations=n_eval, log_dir=log_dir,
                                 eval_env=eval_env, resume=resume,
                                 eval_seeds=[1000 + i for i in range(n_eval)], deterministic=True)

    class SaveVecNormalize(LatestCheckpointCallback):
        def _on_step(self):
            if self.n_calls % self.save_freq == 0 and isinstance(venv, VecNormalize):
                venv.save(vn_path)
            return super()._on_step()

    callbacks = CallbackList([
        eval_cb, ResilienceCallback(),
        CheckpointCallback(save_freq=EVAL_EVERY * update_calls, save_path=os.path.join(log_dir, 'checkpoints'),
                           name_prefix='model'),
        SaveVecNormalize(save_freq=EVAL_EVERY * update_calls, log_dir=log_dir),
    ])
    remaining = time_steps - (model.num_timesteps if resume else 0)
    model.learn(total_timesteps=remaining, callback=callbacks, reset_num_timesteps=not resume)
    model.save(os.path.join(log_dir, 'model_final'))
    if isinstance(venv, VecNormalize):
        venv.save(vn_path)
    venv.close()


if __name__ == '__main__':
    main()
