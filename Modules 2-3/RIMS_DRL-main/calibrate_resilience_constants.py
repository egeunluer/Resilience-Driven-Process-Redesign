#!/usr/bin/env python3
"""
Calibrate and persist the three fixed constants every subsequent simulation on a
given (dataset, resource pool) pair must use, so they are never recomputed live
inside train_DRL.py:

  P_star   : 2 * pooled median(completed-trace cycle times), from N seeded,
             no-shock, least-used-policy baseline episodes (least-used, not
             valid[0]/"FIFO" -- valid[0] concentrates ~30%+ of assignments on
             one alphabetically-first resource and was never the basis for any
             validated result this project has produced).
  cpl_base_table : RF2's CPL_base, one value per shock configuration e_j of
             the training cycle (RF2_reward_function.md SS4) -- mean exact CPL
             area (gym_env.cpl_accumulated) across N seeded random-policy
             episodes under that shock, using P_star above. Keyed by
             shock.shock_key(type, magnitude), e.g. 'DEMAND_SPIKE:2'.
  phi_fix  : RF1's phi_fix -- worst phi(t) observed across K random-policy
             episodes, under the same fixed shock.

No episode run here is time- or trace-limited by this script: every episode
runs to genuine completion (all traces finished, or -- for the shocked runs --
phi(t) recovering back to <= P_star) with no wall-clock cap and no recovery
backstop (gym_env.py's step() no longer has one). The shocked stages
(cpl_base/phi_fix) fire a shock and therefore stop as soon as phi(t) recovers,
so --n-traces for them just needs to be generously large, not tightly sized.
The P_star stage fires no shock at all, so it has nothing to recover from and
always runs to its full trace count -- use --n-traces-baseline to give it a
smaller (but still generously large, pooled-across-seeds) population instead
of paying for a full-length run on every seed just to get a stable median.

Usage:
  python3 calibrate_resilience_constants.py \\
    --dataset BPI_Challenge_2017_W_Two_TS \\
    --artifact "../../Module 1/artifacts/bpi2017.json" \\
    --pool "../../Module 1/artifacts/bpi2017_pure_generalist_pool.json" \\
    --n-traces 5789 --n-traces-baseline 2000 --n-baseline-seeds 8 \\
    --shock-type DEMAND_SPIKE --shock-magnitude 6 --shock-duration 21600 \\
    --cycle-demand-spike 2,3,4,5 --cycle-resource-removal 0.2,0.3,0.4,0.5 \\
    --removal-scope global --n-phi-fix-rollouts 5 \\
    --out "../../Module 1/artifacts/bpi2017_calibration.json"
"""
import argparse
import json
from collections import defaultdict

import numpy as np

from gym_env import gym_env
from shock import ShockConfig, shock_key


def least_used_action(env, valid, assign_count):
    def key(i):
        out = env.output[i]
        resource = out[0] if out != 'Postpone' else '__POSTPONE__'
        return (assign_count[resource], resource)
    return min(valid, key=key)


def random_action(env, valid, assign_count):
    return int(np.random.choice(valid))


def run_episode(env, seed, action_fn):
    """Run one episode to genuine completion. No wall-clock cap, no backstop --
    the caller is responsible for choosing a large enough --n-traces."""
    obs, _ = env.reset(seed=seed)
    assign_count = defaultdict(int)
    terminated = truncated = False
    worst_ppi = 0.0
    while not (terminated or truncated):
        mask = env.action_masks()
        valid = [i for i, m in enumerate(mask) if m]
        if not valid:
            raise RuntimeError(
                f"Empty action mask at seed={seed} -- pool/artifact mismatch, or a "
                f"genuinely uncovered (activity, tier). Fix the pool before calibrating."
            )
        action = action_fn(env, valid, assign_count)
        out = env.output[action]
        resource = out[0] if out != 'Postpone' else '__POSTPONE__'
        assign_count[resource] += 1
        obs, r, terminated, truncated, _ = env.step(action)
        worst_ppi = max(worst_ppi, env._current_ppi())
    return worst_ppi


def build_env(args, reward_function, shock_config, p_star_override=None, phi_fix=None, n_traces=None):
    # reward_function is irrelevant to every stage here (rewards are never read,
    # only phi(t) and the CPL area gym_env tracks for every reward function).
    # Callers pass 'RF1' -- RF2 can't be built here, since it requires the very
    # cpl_base_table this script produces.
    return gym_env(
        args.dataset, n_traces if n_traces is not None else args.n_traces, args.calendar,
        threshold=args.threshold, postpone=False,
        reward_function=reward_function,
        artifact_path=args.artifact,
        shock_config=shock_config,
        pool_override_path=args.pool,
        p_star_override=p_star_override,
        # RF1's reward branch requires phi_fix to already be set -- irrelevant
        # while phi_fix is exactly what's being calibrated here (it isn't read
        # by the phi(t) tracking itself, only by the reward computation, which
        # this calibration loop doesn't use the output of).
        phi_fix=phi_fix if phi_fix is not None else 0.0,
        # D9 (revised): tumbling-batch size for phi(t), not a rolling window --
        # see gym_env.py's ppi_window_n comment. Calibrating P*/cpl_base/phi_fix
        # under a different window size than training later uses would make
        # them meaningless reference points, so this must be threaded through
        # every gym_env construction site consistently (train_DRL.py loads this
        # same value back out of the calibration file this script writes).
        ppi_window_n_override=args.ppi_window_n,
        # Recovery rule for the shocked stages: also wait until every case that
        # arrived during the shock has completed (see gym_env). getattr: other
        # scripts reuse build_env with their own args.
        recovery_requires_backlog_cleared=getattr(args, 'recovery_requires_backlog_cleared', False),
    )


def calibrate_p_star(args):
    # No shock is fired for this stage (shock_config=None), so gym_env.step()'s
    # recovery-based early termination can never trigger here (it requires
    # shock_fired is not None) -- every one of these episodes genuinely runs
    # until its trace population is exhausted, unlike cpl_base/phi_fix below
    # which do fire a shock and therefore do stop early on recovery. Since this
    # stage has nothing to recover from, --n-traces-baseline lets it use a
    # smaller (but still generously large, pooled-across-seeds) population than
    # the full historical trace count, instead of paying for a full run just to
    # get a stable pooled median.
    #
    # --p-star-formula: 'median' (default, P* = 2*pooled median) is the
    # standard definition used everywhere else. 'mean' (P* = 2*pooled mean) is
    # an explicit, dataset-scoped override for logs whose cycle-time
    # distribution is heavily right-skewed/bimodal enough that phi(t) -- a
    # rolling MEAN of completed cycle times -- sits far above 2*median almost
    # continuously even with no shock at all (found on PurchasingExample:
    # baseline phi(t) exceeded a median-based P* ~85-93% of the time across
    # all 8 seeds, with no shock present -- see Modules 2-3/Helpdesk
    # Preprocessing/ session notes... actually see Module 1/ notes for this
    # dataset). Only use 'mean' when a baseline (no-shock) phi(t) trajectory
    # check has confirmed it does NOT leave the system in a permanent
    # near-constant breach state the way the median-based value did.
    n_traces_baseline = args.n_traces_baseline if args.n_traces_baseline is not None else args.n_traces
    print(f"=== P* calibration: {args.n_baseline_seeds} seeded no-shock baseline "
          f"episodes, least-used policy, n_traces={n_traces_baseline}, "
          f"formula={args.p_star_formula} ===", flush=True)
    all_cycle_times = []
    for seed in range(args.n_baseline_seeds):
        env = build_env(args, reward_function='RF1', shock_config=None, n_traces=n_traces_baseline)
        run_episode(env, seed, least_used_action)
        cts = [ct for (_, ct) in env.simulation_process.traces['ended']]
        all_cycle_times.extend(cts)
        print(f"  seed={seed}: n_completed={len(cts)} median={np.median(cts):.1f} "
              f"mean={np.mean(cts):.1f}", flush=True)
    arr = np.array(all_cycle_times)
    median = float(np.median(arr))
    mean = float(np.mean(arr))
    stat = mean if args.p_star_formula == 'mean' else median
    p_star = 2.0 * stat
    print(f"Pooled median={median:.1f}s, pooled mean={mean:.1f}s (n={len(arr)}) -> "
          f"P* = 2*{args.p_star_formula} = {p_star:.1f}s\n", flush=True)
    return p_star, mean


def cycle_schedule_from_args(args):
    """The ordered (shock_type, magnitude) cycle, as train_DRL.py's
    ShockConfig.cycle_schedule must also list it."""
    return ([('DEMAND_SPIKE', float(m)) for m in args.cycle_demand_spike.split(',') if m]
            + [('RESOURCE_REMOVAL', float(m)) for m in args.cycle_resource_removal.split(',') if m])


def calibrate_cpl_base_table(args, p_star):
    # One CPL_base per shock configuration e_j (RF2_reward_function.md SS4): the
    # mean exact CPL area of a non-adaptive (random) policy under that same shock,
    # pooled across n_baseline_seeds seeds so a single non-breaching seed can't
    # zero it out. A value of 0 means that shock never pushed phi(t) above P*
    # under the random policy in any seed -- reported loudly below, since
    # train_DRL.py refuses to normalize RF2's reward by it.
    table, per_seed = {}, {}
    for shock_type, magnitude in cycle_schedule_from_args(args):
        key = shock_key(shock_type, magnitude)
        print(f"=== CPL_base[{key}]: random policy, {args.n_baseline_seeds} seeded episodes, "
              f"duration={args.shock_duration}s, removal scope={args.removal_scope} ===", flush=True)
        shock_config = ShockConfig(
            mode='eval', eval_shock_type=shock_type, eval_magnitude=magnitude,
            eval_duration=args.shock_duration, resource_removal_scope=args.removal_scope,
        )
        cpls = []
        for seed in range(args.n_baseline_seeds):
            env = build_env(args, reward_function='RF1', shock_config=shock_config, p_star_override=p_star)
            run_episode(env, seed=seed, action_fn=random_action)
            cpls.append(float(env.cpl_accumulated))
            print(f"  seed={seed}: CPL = {env.cpl_accumulated:.1f}", flush=True)
        table[key] = float(np.mean(cpls))
        per_seed[key] = cpls
        n_zero = sum(1 for c in cpls if c == 0.0)
        print(f"CPL_base[{key}] = {table[key]:.1f} (mean over {len(cpls)} seeds, {n_zero} never "
              f"breached P*){'  <-- WARNING: zero, unusable as an RF2 normalizer' if table[key] == 0 else ''}\n",
              flush=True)
    return table, per_seed


def calibrate_phi_fix(args, p_star):
    print(f"=== phi_fix calibration: {args.n_phi_fix_rollouts} random-policy rollouts "
          f"under the same fixed shock ===", flush=True)
    shock_config = ShockConfig(
        mode='eval', eval_shock_type=args.shock_type,
        eval_magnitude=args.shock_magnitude, eval_duration=args.shock_duration,
        resource_removal_scope=args.removal_scope,
    )
    worst_per_rollout = []
    for k in range(args.n_phi_fix_rollouts):
        env = build_env(args, reward_function='RF1', shock_config=shock_config, p_star_override=p_star)
        worst = run_episode(env, seed=k, action_fn=random_action)
        worst_per_rollout.append(worst)
        print(f"  rollout {k}: worst phi(t) = {worst:.4f}", flush=True)
    phi_fix = max(worst_per_rollout)
    print(f"phi_fix (worst across {args.n_phi_fix_rollouts} rollouts) = {phi_fix:.4f}\n", flush=True)
    return phi_fix


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--pool', default=None,
                         help="Resource-pool JSON (curated_pt/resource_to_activities) to override the "
                              "artifact's own resource set -- e.g. for the zero-specialist pools built for "
                              "BPI2017/Consulta/Helpdesk. Optional: omit to use the full artifact-driven "
                              "resource set directly (e.g. PurchasingExample, whose resource distribution "
                              "didn't need curating).")
    parser.add_argument('--calendar', action='store_true',
                         help="Use business-hours resource calendars (gym_env's CALENDAR=True) "
                              "instead of always-on availability. Every existing calibration file "
                              "in this project was computed with this OFF (the default here too) -- "
                              "P_star/cpl_base/phi_fix are only valid reference points for the "
                              "calendar setting they were calibrated under, so a calibration file's "
                              "config_used.calendar must match whatever CALENDAR value train_DRL.py/"
                              "compute_resilience_metrics.py run with (both scripts refuse a mismatch).")
    parser.add_argument('--threshold', type=int, default=20)
    parser.add_argument('--n-traces', type=int, required=True,
                         help="Used for the shocked stages (cpl_base/phi_fix), which already "
                              "stop early on recovery -- choose generously, no wall-clock/backstop "
                              "compensates for too small a value")
    parser.add_argument('--n-traces-baseline', type=int, default=None,
                         help="Trace count for the no-shock P* stage specifically, which has "
                              "nothing to recover from and so always runs to full population -- "
                              "defaults to --n-traces if omitted. Set smaller than --n-traces to "
                              "cut P*'s runtime while still pooling a stable median across "
                              "--n-baseline-seeds seeds.")
    parser.add_argument('--n-baseline-seeds', type=int, default=8)
    parser.add_argument('--p-star-formula', choices=['median', 'mean'], default='median',
                         help="P* = 2*pooled median (default, standard) or 2*pooled mean "
                              "(dataset-scoped override for heavily right-skewed/bimodal cycle-time "
                              "distributions -- see calibrate_p_star()'s docstring comment).")
    parser.add_argument('--ppi-window-n', type=int, default=5,
                         help="phi(t) tumbling-batch size: phi(t) is recomputed as the mean of "
                              "every non-overlapping batch of this many new trace completions "
                              "(not a rolling window -- no completion is ever counted in two "
                              "different phi(t) values). Default 5, replacing the old rolling-"
                              "window-of-20 design, which stayed mostly pre-shock data for a long "
                              "stretch after a shock fired. Every gym_env construction on a given "
                              "(dataset, pool) pair -- calibration, training, eval -- must use the "
                              "same value, since P*/cpl_base/phi_fix are only meaningful reference "
                              "points relative to whatever window size computed them.")
    parser.add_argument('--shock-type', default='DEMAND_SPIKE',
                         help="phi_fix stage (RF1) only.")
    parser.add_argument('--shock-magnitude', type=float, required=True,
                         help="phi_fix stage (RF1) only.")
    parser.add_argument('--shock-duration', type=float, required=True,
                         help="Shared by every shocked stage; must match train_DRL.py's duration_range.")
    parser.add_argument('--cycle-demand-spike', default='2,3,4,5',
                         help="DEMAND_SPIKE magnitudes of the training cycle, in order -- one "
                              "CPL_base each. Must match train_DRL.py's cycle_schedule.")
    parser.add_argument('--cycle-resource-removal', default='0.2,0.3,0.4,0.5',
                         help="RESOURCE_REMOVAL magnitudes of the training cycle, in order -- one "
                              "CPL_base each. Must match train_DRL.py's cycle_schedule.")
    parser.add_argument('--recovery-requires-backlog-cleared', action='store_true',
                         help="An episode only counts as recovered once every case that arrived during "
                              "the shock has completed (and phi(t) <= P*). Without it, the first post-shock "
                              "batch of quick cases can end the episode before the shock's backlog -- and "
                              "most of its loss -- is ever measured. Must match train_DRL.py.")
    parser.add_argument('--removal-scope', choices=['global', 'activity'], default='global',
                         help="ShockConfig.resource_removal_scope; must match train_DRL.py.")
    parser.add_argument('--n-phi-fix-rollouts', type=int, default=5)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    p_star, baseline_mean_ct = calibrate_p_star(args)
    cpl_base_table, cpl_base_per_seed = calibrate_cpl_base_table(args, p_star)
    phi_fix = calibrate_phi_fix(args, p_star)

    result = {
        "P_star": p_star,
        # Normal-operation reference for RF2's headroom cap: h_max = P* - mean no-shock
        # cycle time (the pooled mean of the same baseline runs P* comes from).
        "baseline_mean_cycle_time": baseline_mean_ct,
        "h_max": p_star - baseline_mean_ct,
        "cpl_base_table": cpl_base_table,
        "cpl_base_per_seed": cpl_base_per_seed,
        "phi_fix": phi_fix,
        "config_used": {
            "dataset": args.dataset,
            "artifact": args.artifact,
            "pool": args.pool,
            "calendar": args.calendar,
            "threshold": args.threshold,
            "n_traces": args.n_traces,
            "n_traces_baseline": args.n_traces_baseline if args.n_traces_baseline is not None else args.n_traces,
            "n_baseline_seeds": args.n_baseline_seeds,
            "ppi_window_n": args.ppi_window_n,
            "p_star_formula": args.p_star_formula,
            "policy": "least_used",
            "cpl_base_policy": "random",
            "cycle_schedule": [[t, m] for t, m in cycle_schedule_from_args(args)],
            "removal_scope": args.removal_scope,
            "recovery_requires_backlog_cleared": args.recovery_requires_backlog_cleared,
            "shock_type": args.shock_type,
            "shock_magnitude": args.shock_magnitude,
            "shock_duration": args.shock_duration,
            "n_phi_fix_rollouts": args.n_phi_fix_rollouts,
        },
    }
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Wrote calibration to {args.out}", flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
