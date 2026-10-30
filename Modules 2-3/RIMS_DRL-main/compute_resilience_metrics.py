#!/usr/bin/env python3
"""
Compute the four standard resilience metrics -- Time to Impact (TTI), Maximal
Performance Deviation (MPD), Recovery Time (RT), Cumulative Performance Loss
(CPL) -- across a grid of shock magnitudes and seeds, for a (dataset, pool)
pair that already has a persisted calibration file (see
calibrate_resilience_constants.py). Definitions (Training_Readiness_
Implementation_Summary.md SS27):

  TTI : time between shock onset and the first instant phi(t) exceeds P*.
  MPD : the worst (max) phi(t) observed, from shock onset onward.
  RT  : time between first exceeding P* and the next time phi(t) drops back
        to <= P*.
  CPL : the area under max(0, phi(t) - P*), integrated from first-exceed to
        recovery -- a step-function Riemann sum over the logged phi(t)
        trajectory, computed independently here (not read from gym_env's own
        cpl_accumulated) for auditability.

Uses the same fixed P* every other simulation on this (dataset, pool) pair
uses (read from --calibration, never recomputed), the same least-used policy,
and the same no-backstop, recovery-terminated episode structure as
calibrate_resilience_constants.py. If phi(t) never exceeds P* after shock
onset, TTI/RT/CPL are reported as null (undefined) and MPD/breached still
recorded.

Usage:
  python3 compute_resilience_metrics.py \\
    --dataset BPI_Challenge_2017_W_Two_TS \\
    --artifact "../../Module 1/artifacts/bpi2017.json" \\
    --pool "../../Module 1/artifacts/bpi2017_pure_generalist_pool.json" \\
    --calibration "../../Module 1/artifacts/bpi2017_calibration.json" \\
    --n-traces 5789 --magnitudes 5,6,7,8 --duration 21600 --seeds 0,1 \\
    --out "../../Module 1/artifacts/bpi2017_resilience_metrics.json"
"""
import argparse
import json
from collections import defaultdict

import numpy as np

from calibrate_resilience_constants import build_env, least_used_action
from shock import ShockConfig


def run_episode_logged(env, seed, action_fn):
    """Run one episode to genuine completion (recovery or trace exhaustion,
    no backstop), logging (t, phi(t)) at every decision step."""
    obs, _ = env.reset(seed=seed)
    assign_count = defaultdict(int)
    terminated = truncated = False
    trajectory = [(float(env.env.now), env._current_ppi())]
    while not (terminated or truncated):
        mask = env.action_masks()
        valid = [i for i, m in enumerate(mask) if m]
        if not valid:
            raise RuntimeError(f"Empty action mask at seed={seed} -- pool/artifact mismatch.")
        action = action_fn(env, valid, assign_count)
        out = env.output[action]
        resource = out[0] if out != 'Postpone' else '__POSTPONE__'
        assign_count[resource] += 1
        obs, r, terminated, truncated, _ = env.step(action)
        trajectory.append((float(env.env.now), env._current_ppi()))
    return trajectory, env.shock_onset_time, env.shock_offset_time


def compute_metrics(trajectory, shock_onset, p_star):
    post_onset = [(t, phi) for (t, phi) in trajectory if t >= shock_onset]
    if not post_onset:
        return dict(TTI=None, MPD=None, RT=None, CPL=None, breached=False, recovered=None)

    mpd = max(phi for _, phi in post_onset)

    first_exceed_t = next((t for t, phi in post_onset if phi > p_star), None)
    if first_exceed_t is None:
        return dict(TTI=None, MPD=mpd, RT=None, CPL=None, breached=False, recovered=None)

    tti = first_exceed_t - shock_onset

    recovery_t = next(
        (t for t, phi in post_onset if t >= first_exceed_t and phi <= p_star), None
    )
    rt = (recovery_t - first_exceed_t) if recovery_t is not None else None

    window = [
        (t, phi) for (t, phi) in post_onset
        if t >= first_exceed_t and (recovery_t is None or t <= recovery_t)
    ]
    cpl = 0.0
    for i in range(1, len(window)):
        t_prev, phi_prev = window[i - 1]
        t_cur, _ = window[i]
        cpl += max(0.0, phi_prev - p_star) * (t_cur - t_prev)

    return dict(
        TTI=tti, MPD=mpd, RT=rt, CPL=cpl,
        breached=True, recovered=(recovery_t is not None),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--pool', required=True)
    parser.add_argument('--calibration', required=True, help="Output of calibrate_resilience_constants.py")
    parser.add_argument('--calendar', action='store_true',
                         help="Must match the --calendar setting the calibration file was produced "
                              "with (calibrate_resilience_constants.py's config_used.calendar).")
    parser.add_argument('--threshold', type=int, default=20)
    parser.add_argument('--n-traces', type=int, required=True,
                         help="Ceiling only -- shocked episodes terminate on recovery well before this")
    parser.add_argument('--magnitudes', default='5,6,7,8')
    parser.add_argument('--duration', type=float, required=True)
    parser.add_argument('--seeds', default='0,1')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    with open(args.calibration, 'r') as f:
        calibration = json.load(f)
    cfg = calibration.get('config_used', {})
    if cfg.get('dataset') != args.dataset:
        raise ValueError(f"Calibration file computed for dataset '{cfg.get('dataset')}', not '{args.dataset}'.")
    if cfg.get('pool') != args.pool:
        raise ValueError(f"Calibration file computed for pool '{cfg.get('pool')}', not '{args.pool}'.")
    # calendar: see train_DRL.py's identical check -- P_star/cpl_base/phi_fix are
    # only valid reference points for the resource-availability regime (always-on
    # vs. business-hours calendars) they were calibrated under. Calibration files
    # predating this field were all computed with CALENDAR=False, hence the
    # default-to-False fallback rather than skipping the check.
    if cfg.get('calendar', False) != args.calendar:
        raise ValueError(
            f"Calibration file was computed with calendar={cfg.get('calendar', False)}, not "
            f"calendar={args.calendar}. Refusing to reuse a mismatched calibration."
        )
    # Recovery rule (when shocked episodes end): always the one the calibration was
    # computed under, read from the file rather than set separately here.
    args.recovery_requires_backlog_cleared = cfg.get('recovery_requires_backlog_cleared', False)
    p_star = calibration['P_star']
    print(f"Using P_star={p_star:.1f}s from {args.calibration} "
          f"(recovery_requires_backlog_cleared={args.recovery_requires_backlog_cleared})", flush=True)
    # D9 (revised): phi(t)'s tumbling-batch size must match whatever calibrate_
    # resilience_constants.py used to compute P*/cpl_base/phi_fix -- read it back
    # from the same calibration file rather than letting this script default
    # independently. Older calibration files (computed before the rolling-window
    # -> tumbling-batch change) won't have this field; fall back to 20 (the old
    # rolling-window size) with a loud warning, since that calibration is stale
    # under the new phi(t) definition and should be re-run.
    args.ppi_window_n = cfg.get('ppi_window_n')
    if args.ppi_window_n is None:
        args.ppi_window_n = 20
        print(f"WARNING: {args.calibration} has no ppi_window_n recorded (pre-dates the "
              f"tumbling-batch phi(t) redesign) -- falling back to 20. P*/cpl_base/phi_fix in "
              f"this file were calibrated under the OLD rolling-window phi(t), not the new "
              f"tumbling-batch one; re-run calibrate_resilience_constants.py for a consistent "
              f"result.", flush=True)

    magnitudes = [float(m) for m in args.magnitudes.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]

    results = []
    for magnitude in magnitudes:
        shock_config = ShockConfig(
            mode='eval', eval_shock_type='DEMAND_SPIKE',
            eval_magnitude=magnitude, eval_duration=args.duration,
        )
        for seed in seeds:
            print(f"=== magnitude={magnitude} seed={seed} ===", flush=True)
            env = build_env(args, reward_function='RF1', shock_config=shock_config, p_star_override=p_star)
            trajectory, onset, offset = run_episode_logged(env, seed, least_used_action)
            metrics = compute_metrics(trajectory, onset, p_star)
            metrics.update(magnitude=magnitude, seed=seed, shock_onset=onset, shock_offset=offset,
                            n_completed=len(env.simulation_process.traces['ended']))
            print(f"  TTI={metrics['TTI']} MPD={metrics['MPD']} RT={metrics['RT']} "
                  f"CPL={metrics['CPL']} recovered={metrics['recovered']}", flush=True)
            results.append(metrics)

    out = {
        "P_star": p_star,
        "config_used": {
            "dataset": args.dataset, "artifact": args.artifact, "pool": args.pool,
            "calibration": args.calibration, "calendar": args.calendar, "threshold": args.threshold,
            "recovery_requires_backlog_cleared": args.recovery_requires_backlog_cleared,
            "n_traces": args.n_traces, "magnitudes": magnitudes, "duration": args.duration,
            "seeds": seeds, "policy": "least_used", "shock_type": "DEMAND_SPIKE",
        },
        "results": results,
    }
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Wrote resilience metrics to {args.out}", flush=True)


if __name__ == '__main__':
    main()
