"""
MaskablePPO with per-case episodes for loan_env.LoanEnv(reward='case').

The simulation still runs one disruption per environment episode, with the decisions
of all cases in time order. For learning, each case's decisions form their own short
episode (chain), with the case's own Loss share as reward (see loan_env). That share is
only known once the disruption episode is over, so rollouts are collected in whole
episodes:

  - every environment step is kept per environment (observation, action, mask,
    immediate reward) until its episode ends; then info['case_trace'] gives each step's
    case id and delayed reward, and the steps are regrouped into case chains;
  - a rollout ends once at least n_steps * n_envs steps of finished episodes are ready
    (episodes still running carry over to the next rollout);
  - the buffer holds the chains back to back (one environment column, an episode start
    at every chain start, the last step of a chain terminal); values and log-probs are
    recomputed with the current policy, since carried-over steps were sampled before
    the last update; then the usual GAE and PPO update.
"""
import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.utils import obs_as_tensor


class CaseMaskablePPO(MaskablePPO):
    def _setup_model(self):
        super()._setup_model()
        self._open_steps = None     # per environment: steps of the running episode
        self._ready = []            # finished chains: lists of (obs, action, mask, reward)
        self._n_ready = 0

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ['_open_steps', '_ready', '_n_ready']

    def _finish_episode(self, steps, trace):
        if len(trace['cid']) != len(steps):
            raise RuntimeError(f"case_trace has {len(trace['cid'])} steps, the rollout {len(steps)}")
        chains = {}
        for (obs, a, m, r), cid, late in zip(steps, trace['cid'], trace['delayed']):
            chains.setdefault(cid, []).append((obs, a, m, r + late))
        self._ready.extend(chains.values())
        self._n_ready += len(steps)

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps, use_masking=True):
        assert self._last_obs is not None, 'No previous observation was provided'
        self.policy.set_training_mode(False)
        if self._open_steps is None:
            self._open_steps = [[] for _ in range(env.num_envs)]
        target = n_rollout_steps * env.num_envs
        callback.on_rollout_start()

        while self._n_ready < target:
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                action_masks = get_action_masks(env)
                actions, values, log_probs = self.policy(obs_tensor, action_masks=action_masks)
            actions = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions)
            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)

            for i in range(env.num_envs):
                self._open_steps[i].append((self._last_obs[i], int(actions[i]), action_masks[i], float(rewards[i])))
                if dones[i]:
                    if 'case_trace' not in infos[i]:
                        raise RuntimeError("CaseMaskablePPO needs LoanEnv(reward='case')")
                    self._finish_episode(self._open_steps[i], infos[i]['case_trace'])
                    self._open_steps[i] = []
            self._last_obs = new_obs
            self._last_episode_starts = dones

        self.rollout_buffer = self._fill_buffer()
        callback.on_rollout_end()
        return True

    def _fill_buffer(self):
        chains, self._ready, n, self._n_ready = self._ready, [], self._n_ready, 0
        buf = MaskableRolloutBuffer(n, self.observation_space, self.action_space, device=self.device,
                                    gae_lambda=self.gae_lambda, gamma=self.gamma, n_envs=1)
        steps = [s for ch in chains for s in ch]
        obs = np.array([s[0] for s in steps], dtype=np.float32)
        acts = np.array([s[1] for s in steps])
        masks = np.array([s[2] for s in steps], dtype=np.float32)
        starts = np.zeros(n, dtype=np.float32)
        starts[np.cumsum([0] + [len(ch) for ch in chains[:-1]])] = 1.0
        with th.no_grad():
            values, log_probs = [], []
            for lo in range(0, n, 4096):
                v, lp, _ = self.policy.evaluate_actions(obs_as_tensor(obs[lo:lo + 4096], self.device),
                                                        th.as_tensor(acts[lo:lo + 4096], device=self.device),
                                                        action_masks=masks[lo:lo + 4096])
                values.append(v.flatten().cpu().numpy())
                log_probs.append(lp.cpu().numpy())
        buf.observations[:] = obs.reshape(buf.observations.shape)
        buf.actions[:] = acts.reshape(buf.actions.shape)
        buf.action_masks[:] = masks.reshape(buf.action_masks.shape)
        buf.rewards[:] = np.array([s[3] for s in steps], dtype=np.float32).reshape(-1, 1)
        buf.episode_starts[:] = starts.reshape(-1, 1)
        buf.values[:] = np.concatenate(values).reshape(-1, 1)
        buf.log_probs[:] = np.concatenate(log_probs).reshape(-1, 1)
        buf.pos, buf.full = n, True
        # every chain ends terminal: the step after the buffer's last is a new chain
        buf.compute_returns_and_advantage(last_values=th.zeros(1), dones=np.ones(1))
        returns = [sum(s[3] for s in ch) for ch in chains]
        self.logger.record('rollout/case_chains', len(chains))
        self.logger.record('rollout/case_chain_len', n / max(1, len(chains)))
        self.logger.record('rollout/case_return_mean', float(np.mean(returns)))
        return buf
