import os
import json

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import TD3
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

from typing import Callable

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-5:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True

class EvalPolicyCallback(BaseCallback):
    def __init__(self, check_freq: int, nr_evaluations: int, log_dir: str, eval_env, model_name: str = 'best_model', verbose: int = 1, resume: bool = False):
        super(EvalPolicyCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.nr_evaluations = nr_evaluations
        self.log_dir = log_dir
        self.eval_env = eval_env
        self.save_path = os.path.join(log_dir, model_name)
        self.best_mean_cycle_time = np.inf
        self.best_mean_cpl_ratio = np.inf
        self.prev_steps = 0
        self.episode_lengths = []
        # Best score so far, persisted next to best_model.zip. On a resumed run it
        # is read back, so the first evaluation after resuming can't overwrite a
        # better best_model just because the in-memory best started at infinity.
        # Ignored on a fresh run, so a leftover file from an earlier run in the
        # same folder never carries over.
        self.best_score_path = os.path.join(log_dir, 'best_model_score.json')
        if resume and os.path.exists(self.best_score_path):
            with open(self.best_score_path) as f:
                best = json.load(f)
            self.best_mean_cpl_ratio = best.get('best_mean_cpl_ratio', np.inf)
            self.best_mean_cycle_time = best.get('best_mean_cycle_time', np.inf)
            print(f"Resumed best-model score from {self.best_score_path}: {best}", flush=True)

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if (self.n_calls - 1) % self.check_freq == 0 and self.n_calls > 1: # After model update
            print('\n-------- Evaluating current policy --------')
            mean_cycle_times = []
            total_rewards = []
            eval_metrics = {}
            # eval_env may be wrapped (e.g. Monitor in train_DRL.py); gymnasium 1.x
            # wrappers no longer forward attribute lookups, so action_masks()/
            # mean_cycle_time must be read from the underlying gym_env.
            raw_env = self.eval_env.unwrapped
            for epoch in range(self.nr_evaluations):
                state, _ = self.eval_env.reset()
                isTerminated = False

                while isTerminated == False:
                    action, _state = self.model.predict(state, action_masks=raw_env.action_masks())
                    state, reward, isTerminated, _, _ = self.eval_env.step(action)

                mean_cycle_times.append(raw_env.mean_cycle_time)
                total_rewards.append(raw_env.mean_cycle_time)
                for key, value in raw_env.last_episode_info.items():
                    if key != 'shock':
                        eval_metrics.setdefault(key, []).append(value)

            mean_cycle_time = sum(mean_cycle_times)/len(mean_cycle_times)
            total_reward = sum(total_rewards)/len(total_rewards)

            # TensorBoard: mean resilience metrics over this evaluation round (one
            # full shock cycle when a cycle_schedule is set). Written at the next
            # logger dump, i.e. alongside the following rollout's train/ scalars.
            self.logger.record('eval/mean_cycle_time_hours', mean_cycle_time / 3600)
            for key, values in eval_metrics.items():
                self.logger.record(f'eval/{key}', float(np.mean(values)))

            print(f"Num timesteps: {self.num_timesteps}")
            # Best-model selection: lowest mean CPL ratio (B_K / CPL_base[e_j], RF2's
            # objective) when every evaluation episode reports one; otherwise (reward
            # functions without a CPL_base table) the original lowest mean cycle time.
            cpl_ratios = eval_metrics.get('cpl_ratio', [])
            if len(cpl_ratios) == self.nr_evaluations:
                mean_cpl_ratio = float(np.mean(cpl_ratios))
                print(f"Best mean CPL ratio: {self.best_mean_cpl_ratio:.4f} - Last mean CPL ratio: {mean_cpl_ratio:.4f}")
                is_best = mean_cpl_ratio < self.best_mean_cpl_ratio
                if is_best:
                    self.best_mean_cpl_ratio = mean_cpl_ratio
            else:
                print(f"Best mean cycle time: {self.best_mean_cycle_time:.2f} - Last mean cycle time: {mean_cycle_time:.2f}")
                is_best = mean_cycle_time < self.best_mean_cycle_time
                if is_best:
                    self.best_mean_cycle_time = mean_cycle_time

            if is_best:
                if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                self.model.save(self.save_path)
                with open(self.best_score_path, 'w') as f:
                    json.dump({'best_mean_cpl_ratio': self.best_mean_cpl_ratio,
                               'best_mean_cycle_time': self.best_mean_cycle_time,
                               'num_timesteps': self.num_timesteps}, f, indent=2)
            print('---- Finished evaluating current policy ----\n')
        return True


class LatestCheckpointCallback(BaseCallback):
    """
    Saves the current model (policy + optimizer state + num_timesteps) to
    <log_dir>/latest_model.zip every `save_freq` steps, overwriting the previous
    one, so a run interrupted mid-way (e.g. a Colab disconnect) can continue from
    its latest state via train_DRL.py's RIMS_RESUME, not only from best_model.
    Written to a temporary file first and then renamed, so an interruption during
    the save itself can't leave a half-written latest_model.zip behind.
    """
    def __init__(self, save_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.path = os.path.join(log_dir, 'latest_model.zip')

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            tmp = self.path[:-len('.zip')] + '_tmp.zip'
            self.model.save(tmp)
            os.replace(tmp, self.path)
            if self.verbose >= 1:
                print(f"Checkpoint: saved latest model at {self.num_timesteps} steps to {self.path}", flush=True)
        return True


class ResilienceCallback(BaseCallback):
    """
    Logs gym_env's episode-end resilience metrics (info['resilience'], see
    gym_env._resilience_info) to TensorBoard during training, so CPL ratio,
    recovery and deviation are visible next to the reward. Each metric is
    averaged over the episodes that ended within one rollout, both across all
    shocks (resilience/...) and per shock configuration (resilience_<shock>/...),
    since a cycle of magnitudes would otherwise show up as a sawtooth.
    """
    def _on_step(self) -> bool:
        for info, done in zip(self.locals['infos'], self.locals['dones']):
            metrics = info.get('resilience') if done else None
            if not metrics:
                continue
            shock_tag = metrics['shock'].replace(':', '_')
            for key, value in metrics.items():
                if key == 'shock':
                    continue
                self.logger.record_mean(f'resilience/{key}', value)
                self.logger.record_mean(f'resilience_{shock_tag}/{key}', value)
        return True


def custom_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        if progress_remaining > 0.8: #0.95
            return initial_value
        elif progress_remaining > 0.5: #0.9
            return initial_value * 0.5
        else:
            return initial_value * 0.1

    return func

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func