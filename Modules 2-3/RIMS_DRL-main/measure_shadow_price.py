#!/usr/bin/env python3
"""
Shadow price of the escalation backup workers, measured from the process.

The per-case reward cannot see what an escalation costs the other cases: while a
backup worker handles one case, it is not free for the next one. That cost is
measured here, per backup team (one per failure type), under a given policy (the
current agent): every episode is run with the normal team (team_per_item workers
per step) and with one worker fewer, on the same seeds. The smaller team forgoes
some escalations; the extra Loss of the process divided by the number of
escalations forgone is the value of one escalation's worker time to the rest of
the process:

    price[cause] = (Loss_small - Loss_full) / Loss_base * N / (esc_full - esc_small)

in per-decision reward units (the units of kappa). loan_env charges it to every
escalation of that team (reward only; the objective already holds the real cost in
the Loss). It is the same for every escalation of a team: it never looks at the
escalated case's own urgency, so the agent still has to learn when an escalation
is worth its price. It depends on the policy that uses the workers, so it can be
re-measured after training with it.

  python3 measure_shadow_price.py --agent tmp_loan_training/<run>/model_final.zip
  python3 measure_shadow_price.py --agent ... --write    # into the calibration file
"""
import argparse
import copy
import json
import os
from multiprocessing import Pool

import numpy as np

ART = '../../Module 1/artifacts/'
PATHS = (ART + 'bpic2017_stage_model.json', ART + 'bpic2017_disruptions.json',
         ART + 'bpic2017_pool3_resilience_calibration.json')
_G = {}


def _init(agent):
    import torch
    from sb3_contrib import MaskablePPO
    torch.set_num_threads(1)
    _G['model'] = MaskablePPO.load(agent, device='cpu')


def _job(args):
    from loan_env import LoanEnv
    scenario, seed, team = args
    cfg = copy.deepcopy(json.load(open(PATHS[1])))
    cfg['recovery']['escalate']['team_per_item'] = team
    env = LoanEnv(PATHS[0], cfg, PATHS[2], cycle=[(scenario, '2', None)], reward='global')
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        a, _ = _G['model'].predict(obs, action_masks=env.action_masks(), deterministic=True)
        obs, _, done, _, info = env.step(int(a))
    res = env.ep.result()
    return scenario, seed, team, res['loss'], res['actions'].get('escalate', 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--agent', required=True, help='MaskablePPO model whose escalations are priced')
    ap.add_argument('--seeds', type=int, default=16)
    ap.add_argument('--seed0', type=int, default=300)
    ap.add_argument('--write', action='store_true', help='store the prices in the calibration file')
    args = ap.parse_args()
    cfg = json.load(open(PATHS[1]))
    cal = json.load(open(PATHS[2]))
    full = cfg['recovery']['escalate']['team_per_item']
    cells = [tuple(c) for c in cfg['training_cycle']['cells']]
    seeds = range(args.seed0, args.seed0 + args.seeds)
    jobs = [(c[0], s, t) for c in cells for s in seeds for t in (full, full - 1)]
    with Pool(max(1, (os.cpu_count() or 2) - 1), initializer=_init, initargs=(args.agent,)) as pool:
        R = pool.map(_job, jobs, chunksize=1)
    by = {(sc, s, t): (loss, esc) for sc, s, t, loss, esc in R}
    prices = {}
    print(f'teams of {full} vs {full - 1} backup workers per step, {args.seeds} seeds, agent {args.agent}')
    for sc, lv, _ in cells:
        n = cal['loss_base_by_scenario'][f'{sc}:{lv}:cycle']
        cause = cfg['scenarios'][sc]['failure_type']
        dl = np.array([(by[(sc, s, full - 1)][0] - by[(sc, s, full)][0]) / n['loss_base'] * n['n_decisions'] for s in seeds])
        de = np.array([by[(sc, s, full)][1] - by[(sc, s, full - 1)][1] for s in seeds])
        price = dl.sum() / de.sum() if de.sum() > 0 else 0.0
        # interval by resampling seeds
        rng = np.random.default_rng(0)
        boot = [dl[i].sum() / max(1, de[i].sum()) for i in (rng.integers(len(dl), size=len(dl)) for _ in range(2000))]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        prices[cause] = max(0.0, float(price))
        print(f'  {sc:22s} ({cause}): escalations forgone {de.mean():5.1f}/episode, extra Loss {dl.mean():+7.2f} '
              f'reward units/episode -> price {price:+.3f} per escalation (95% {lo:+.3f}..{hi:+.3f})')
    if args.write:
        cal['escalation_shadow_price'] = prices
        cal['escalation_shadow_price_definition'] = (
            f'per escalation, per-decision reward units: extra Loss of the process per escalation forgone when the '
            f'backup team shrinks from {full} to {full - 1} workers per step, under {os.path.basename(args.agent)} '
            f'({args.agent}), seeds {seeds[0]}-{seeds[-1]}; negative estimates set to 0')
        json.dump(cal, open(PATHS[2], 'w'), indent=1)
        print(f'-> {PATHS[2]}')


if __name__ == '__main__':
    main()
