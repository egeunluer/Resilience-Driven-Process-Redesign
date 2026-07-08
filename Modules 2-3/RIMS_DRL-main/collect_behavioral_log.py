"""
Step 19 — Post-training behavioral log collector.

Loads a frozen trained MaskablePPO model, runs it through N_EPISODES shock-injected
simulation episodes, and writes a flat CSV that Module 4 consumes for distillation.

Each row in the CSV is one non-Postpone decision the agent made:
  - pending_activity : which activity was waiting for a resource
  - shock_active     : 1 if a shock was active at that moment, 0 otherwise
  - tier_chosen      : the performance tier of the resource the agent picked
                       (the distillation label)
  - <act>|<tier>     : one column per (activity, tier) pair found in resource_tier_map,
                       value = number of available resources with that tier for that activity
                       (the distillation features / cluster-layer observation)

Why post-training only:
  During training the policy changes every update. Collecting then mixes decisions
  from many different policy versions. Module 4 needs the behavioral log of the
  *final* policy, so collection must happen after training with the frozen model.

Usage:
  python collect_behavioral_log.py \\
      --name-log     bpi12w \\
      --n-traces     500 \\
      --artifact     "../Module 1/artifacts/bpi12w.json" \\
      --model        "tmp_training_2/.../model_final" \\
      --output       "../Module 1/artifacts/bpi12w_behavioral_log.csv" \\
      --n-episodes   50
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser(description="Collect behavioral log from a frozen DRL policy.")
    p.add_argument("--name-log",   required=True,  help="Dataset name (matches example/ folder)")
    p.add_argument("--n-traces",   default="from_input_data",
                   help="Number of traces per episode, or 'from_input_data'")
    p.add_argument("--artifact",   required=True,  help="Path to Module 1 JSON artifact")
    p.add_argument("--model",      required=True,  help="Path to saved MaskablePPO model (no .zip extension)")
    p.add_argument("--output",     required=True,  help="Path for output CSV file")
    p.add_argument("--n-episodes", type=int, default=50,
                   help="Number of collection episodes (default: 50)")
    p.add_argument("--shock",      action="store_true",
                   help="Inject shocks during collection (recommended: same config as training)")
    p.add_argument("--threshold",  type=int, default=0)
    return p.parse_args()


def build_shock_config():
    """Returns the same ShockConfig used during training. Edit here if you changed training config."""
    from shock import ShockConfig
    return ShockConfig(
        mode='eval',              # single controlled shock per episode for clean collection
        eval_onset_time=3600.0,
        eval_shock_type='RESOURCE_REMOVAL',
        eval_magnitude=0.3,
        eval_duration=3600.0,
    )


def collect(args):
    from gym_env import gym_env
    from sb3_contrib import MaskablePPO

    n_traces = args.n_traces if args.n_traces == 'from_input_data' else int(args.n_traces)
    shock_config = build_shock_config() if args.shock else None

    # Build the environment — same constructor as training, CALENDAR=False (D10).
    env = gym_env(
        args.name_log,
        n_traces,
        CALENDAR=False,
        postpone=False,          # no Postpone during collection — every step is a real decision
        reward_function='inverse_CT',   # reward is irrelevant during collection
        artifact_path=args.artifact,
        shock_config=shock_config,
        threshold=args.threshold,
    )

    model = MaskablePPO.load(args.model, env=env)
    print(f"Model loaded from {args.model}", flush=True)

    # Discover the full set of (activity, tier) keys across all resources so every
    # row has the same columns regardless of which pairs appear in a given episode.
    all_cluster_keys = sorted({
        (activity, tier)
        for resource, activity_tiers in env.resource_tier_map.items()
        for activity, tier in activity_tiers.items()
    })
    col_names = [f"{act}|{tier}" for act, tier in all_cluster_keys]

    rows = []
    total_decisions = 0

    for episode in range(args.n_episodes):
        obs, _ = env.reset()
        done = False
        step_in_ep = 0

        while not done:
            mask = env.action_masks()
            action_idx, _ = model.predict(obs, deterministic=True, action_masks=mask)
            action_idx = int(action_idx)

            chosen = env.output[action_idx]

            # Only log real assignments — Postpone carries no tier information.
            if chosen != 'Postpone':
                resource_chosen, activity_chosen = chosen

                # Translate agent's individual-level choice to a tier (cluster-layer action).
                tier_chosen = env.resource_tier_map.get(resource_chosen, {}).get(activity_chosen, 'unknown')

                # Cluster-layer observation: available resource counts per (activity, tier).
                cluster_obs = env.get_cluster_obs()

                row = {
                    'episode':          episode,
                    'step':             step_in_ep,
                    'pending_activity': activity_chosen,
                    'shock_active':     int(env.simulation_process.shock_state.is_active),
                    'tier_chosen':      tier_chosen,
                }
                # Flatten cluster obs into fixed columns (0 for pairs not present this step).
                for key, col in zip(all_cluster_keys, col_names):
                    row[col] = cluster_obs.get(key, 0)

                rows.append(row)
                total_decisions += 1

            obs, _reward, done, _, _ = env.step(action_idx)
            step_in_ep += 1

        print(f"Episode {episode + 1}/{args.n_episodes} — {step_in_ep} steps", flush=True)

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"\nDone. {total_decisions} decisions across {args.n_episodes} episodes.", flush=True)
    print(f"Behavioral log written to: {args.output}", flush=True)
    print(f"Columns: {list(df.columns)}", flush=True)
    print(f"Tier distribution:\n{df['tier_chosen'].value_counts()}", flush=True)


if __name__ == '__main__':
    collect(parse_args())
