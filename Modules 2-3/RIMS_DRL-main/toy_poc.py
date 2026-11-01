"""
Phase 1 proof of concept: train MaskablePPO on loan_model.toy and compare it with
the baselines on paired seeds (drl_resilience_design_decisions.md §16).

  python3 toy_poc.py                # calibrate, train, select, benchmark
  RIMS_TIME_STEPS=300000 python3 toy_poc.py

Seeds: calibration 0-15, checkpoint selection 1000-1063, benchmark 200-455 -- three
disjoint sets, so the reported benchmark isn't the set the model was picked on.
Pass criterion: PPO beats always_retry, always_escalate and greedy with a 95%
interval of the paired return difference above zero.
"""
import os
import json
import glob

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from loan_model.toy import ToyTriageEnv, BASELINES, run_episode

LOG_DIR = os.path.join(os.environ.get('RIMS_LOG_ROOT', 'tmp_training_2'), 'toy_poc' + (
    '_' + os.environ['RIMS_RUN_TAG'] if os.environ.get('RIMS_RUN_TAG') else ''))
TIME_STEPS = int(os.environ.get('RIMS_TIME_STEPS', 1000000))
ENT_COEF = float(os.environ.get('TOY_ENT_COEF', 0.01))
N_STEPS = int(os.environ.get('TOY_N_STEPS', 4096))
NORM_REWARD = os.environ.get('TOY_NORM_REWARD', '1') == '1'
TRAIN_SEED = int(os.environ.get('TOY_TRAIN_SEED', 0))
CALIB_SEEDS = range(0, 16)
SELECT_SEEDS = range(1000, 1064)
BENCH_SEEDS = range(200, 456)


def evaluate(policy, seeds, loss_norm):
    env = ToyTriageEnv(loss_norm=loss_norm)
    return [run_episode(env, policy, s) for s in seeds]


def paired(a, b):
    d = np.array([x[0] - y[0] for x, y in zip(a, b)])
    return d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(len(d))


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    # Loss normalisation: mean do-nothing Loss (the analogue of CPL_base).
    calib = evaluate(BASELINES['do_nothing'], CALIB_SEEDS, 1.0)
    loss_norm = float(np.mean([i['loss'] for _, i in calib]))
    print(f'loss_norm (do-nothing mean Loss, seeds 0-15): {loss_norm:.3f}', flush=True)

    # Entropy bonus and reward normalisation: without them the first run collapsed
    # to do-nothing within ~150k steps (every reward mixes all 30 cases' deadline
    # misses, so one decision's effect is small against the episode noise).
    env = ToyTriageEnv(loss_norm=loss_norm)
    if NORM_REWARD:
        env = VecNormalize(DummyVecEnv([lambda: ToyTriageEnv(loss_norm=loss_norm)]),
                           norm_obs=False, norm_reward=True, gamma=1.0)
    model = MaskablePPO('MlpPolicy', env, learning_rate=3e-4, n_steps=N_STEPS, batch_size=256,
                        gamma=1.0, gae_lambda=0.95, ent_coef=ENT_COEF, verbose=0, seed=TRAIN_SEED)
    model.set_logger(configure(LOG_DIR, ['csv', 'tensorboard']))
    model.learn(total_timesteps=TIME_STEPS,
                callback=CheckpointCallback(save_freq=50000, save_path=LOG_DIR, name_prefix='ckpt'))
    model.save(os.path.join(LOG_DIR, 'model_final'))

    # Checkpoint selection on its own seed set (every checkpoint kept).
    ckpts = sorted(glob.glob(os.path.join(LOG_DIR, 'ckpt_*_steps.zip')),
                   key=lambda p: int(p.split('_')[-2])) + [os.path.join(LOG_DIR, 'model_final.zip')]
    scores = {}
    for p in ckpts:
        res = evaluate(MaskablePPO.load(p), SELECT_SEEDS, loss_norm)
        scores[os.path.basename(p)] = float(np.mean([r for r, _ in res]))
        print(f'  select {os.path.basename(p):28s} mean return {scores[os.path.basename(p)]:7.3f}', flush=True)
    best = max(scores, key=scores.get)
    print(f'selected checkpoint: {best}', flush=True)

    # Benchmark on disjoint seeds.
    results = {name: evaluate(pol, BENCH_SEEDS, loss_norm) for name, pol in BASELINES.items()}
    results['ppo'] = evaluate(MaskablePPO.load(os.path.join(LOG_DIR, best)), BENCH_SEEDS, loss_norm)
    results['ppo_final'] = evaluate(MaskablePPO.load(os.path.join(LOG_DIR, 'model_final.zip')), BENCH_SEEDS, loss_norm)

    print(f'\nBenchmark: {len(BENCH_SEEDS)} paired episodes (seeds {BENCH_SEEDS.start}-{BENCH_SEEDS.stop - 1})')
    print(f"{'policy':18s} {'return':>8s} {'loss ratio':>10s} {'missed':>7s} {'act.cost':>8s}  "
          f"{'retry':>6s} {'subst':>6s} {'escal':>6s} {'wait':>6s}")
    for name, res in results.items():
        inf = [i for _, i in res]
        m = lambda k: np.mean([i[k] for i in inf])
        print(f"{name:18s} {np.mean([r for r, _ in res]):8.3f} {m('loss_ratio'):10.3f} {m('missed'):7.2f} "
              f"{m('action_cost'):8.2f}  {m('n_retry'):6.1f} {m('n_substitute'):6.1f} {m('n_escalate'):6.1f} {m('n_do_nothing'):6.1f}")
    print('\nPaired return difference, PPO (selected) - baseline, mean +- 95% CI:')
    summary = {'loss_norm': loss_norm, 'selected': best, 'diffs': {}}
    for name in BASELINES:
        d, ci = paired(results['ppo'], results[name])
        summary['diffs'][name] = [d, ci]
        print(f'  vs {name:18s} {d:+.3f} +- {ci:.3f}  {"BETTER" if d - ci > 0 else ("worse" if d + ci < 0 else "n.s.")}')
    passed = all(summary['diffs'][n][0] - summary['diffs'][n][1] > 0 for n in ('always_retry', 'always_escalate', 'greedy'))
    summary['passed'] = passed
    print(f'\nPhase 1 pass criterion (beats always_retry, always_escalate, greedy): {"PASSED" if passed else "NOT PASSED"}')
    json.dump(summary, open(os.path.join(LOG_DIR, 'benchmark_summary.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
