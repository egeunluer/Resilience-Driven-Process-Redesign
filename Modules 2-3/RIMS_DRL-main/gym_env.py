import gymnasium as gym
from gymnasium import spaces, Env
import numpy as np
import random
from typing import List
import csv
import simpy
import utility
from process import SimulationProcess
from event_trace import Token
from parameters import Parameters
from role_simulator import RoleSimulator
import sys, getopt
from utility import *
import pm4py
from inter_trigger_timer import InterTriggerTimer
from shock import fire_single_shock
import warnings
import json
import math
import pm4py
from os.path import exists
from datetime import datetime, timedelta
import random
CYCLE_TIME_MAX = 8.64e+6

# Synthetic resource for zero-eligible-resource activities (Module 1 activities_at_risk).
# Always available (infinite capacity), never entered into resource_tier_map — a
# deadlock-avoidance placeholder, not a resource meant to be genuinely selected between.
PLACEHOLDER_RESOURCE = '__NO_RESOURCE__'
_ALWAYS_ON_CALENDAR = {'days': [0, 1, 2, 3, 4, 5, 6], 'hour_min': 0, 'hour_max': 24}


DEBUG_PRINT = False
if __name__ == "__main__":
    warnings.filterwarnings("ignore")


class gym_env(Env):
    def __init__(self, NAME_LOG, N_TRACES, CALENDAR, POLICY=None, N_SIMULATION=0, threshold=0, postpone=True, reward_function='inverse_CT', artifact_path=None, shock_config=None, cpl_budget=0.0, eta_lambda=0.01, phi_fix=None, beta=0.5, pool_override_path=None, p_star_override=None, ppi_window_n_override=None) -> None:
        self.name_log = NAME_LOG

        self.CALENDAR = CALENDAR ## "True" If you want to use calendar, "False" otherwise
        self.threshold = threshold
        self.postpone = postpone
        self.reward_function = reward_function # 'inverse_CT' or 'negative_CT'

        self.total_reward = 0
        self.reward_count = 0

        self.normalization_cycle_time = 10000#= self.normalization_cycle_times[self.name_log] if normalization else 0
        #self.median_processing_time = retrieve_median_processing_time(NAME_LOG)
        self.policy = POLICY
        self.print = False
        if threshold > 0:
            input_file = './example/' + self.name_log + '/input_' + self.name_log + str(threshold) + '.json'
        else:
            input_file = './example/' + self.name_log + '/input_' + self.name_log + '.json'
        with open(input_file, 'r') as f:
            input_data = json.load(f)

        # Load Module 1 empirical artifact (D1, D4).
        # artifact_path must point to the JSON produced by empirical_preprocessing.py.
        if artifact_path is None:
            raise ValueError(
                "artifact_path is required. Pass the path to the Module 1 JSON artifact "
                "(e.g. '../Module 1/artifacts/bpi12w.json')."
            )
        with open(artifact_path, 'r') as f:
            artifact = json.load(f)
        self.resource_processing_times = artifact['resource_processing_times']
        self.resource_tier_map = artifact['resource_tier_map']
        self.activities_at_risk = artifact.get('activities_at_risk', [])
        # D-merge-fix: some datasets (e.g. Consulta) merge two raw Petri-net
        # activity labels into one Module 1 activity for statistics purposes
        # (activity_merge_map, Phase 9.6). The mined .pnml keeps the two
        # original transitions, so tokens still ask for the native label —
        # this map translates native -> merged wherever a token's activity is
        # compared against the (Module-1-derived) action/task-type space.
        # Empty for datasets that don't merge anything (no-op).
        self.activity_merge_map = artifact.get('config_used', {}).get('activity_merge_map') or {}
        # D1/D9: P*, per-dataset normalization constant (C), and the PPI rolling
        # window size, all sourced from the Module 1 artifact (Phase 9).
        # p_star_override: a fixed P* computed once by calibrate_resilience_constants.py
        # for a specific (dataset, pool) pair, held constant across every subsequent
        # calibration/training/eval run on that pair. Takes priority over the
        # artifact's own P_star (which is still the historical p95, not this
        # simulator-native value) when given.
        self.P_star = p_star_override if p_star_override is not None else artifact.get('P_star')
        self.normalization_cycle_time = artifact.get('normalization_cycle_time', self.normalization_cycle_time)
        # D9 (revised): ppi_window_n is now the size of a non-overlapping ("tumbling")
        # batch, not a rolling/sliding window. phi(t) is recomputed only once every
        # ppi_window_n NEW trace completions, as the mean of exactly that batch -- no
        # completion is ever counted in two different phi(t) values. This replaced an
        # earlier rolling-mean-of-last-20 design, which stayed ~80% pre-shock data for
        # a long stretch after a shock fired (each new completion only evicted the
        # single oldest one from a 20-slot window), diluting the shock's visibility in
        # phi(t) for many steps. ppi_window_n_override lets callers use a smaller batch
        # (e.g. 5) than whatever value is on the artifact (every artifact currently
        # says 20, inherited from the old rolling-window design) without editing the
        # artifact file itself -- same override pattern as p_star_override above.
        self.ppi_window_n = ppi_window_n_override if ppi_window_n_override is not None else artifact.get('ppi_window_n', 20)

        self.shock_config = shock_config  # D5: ShockConfig instance or None
        # D-single-shock: single-shock-per-episode bookkeeping (only used when
        # shock_config is not None). shock_fired/onset/offset are set fresh every
        # reset(). No time-based recovery backstop: an episode ends only via
        # genuine recovery (phi(t) <= P_star after the shock ends) or the trace
        # population running out — a fixed 5x-duration backstop was found to
        # truncate episodes well before real recovery happened (confirmed on a
        # Consulta run that reported as truncated at the backstop but genuinely
        # recovered at 7.3 days once the backstop was removed).
        self.shock_fired = None
        self.shock_onset_time = None
        self.shock_offset_time = None
        # D6 (RF2): constrained-MDP parameters.
        self.cpl_budget = cpl_budget       # CPL_max = p * CPL_base (p applied by the caller)
        self.eta_lambda = eta_lambda       # Lagrange multiplier step size
        self.lagrange_lambda = 0.0         # lambda_k — persists across episodes, updated at episode end
        # D9 (RF1): potential-based shaping parameters.
        self.phi_fix = phi_fix             # worst PPI from random-policy calibration rollouts
        self.beta = beta                   # r1/r2 convex-combination weight, beta in (0,1)

        self.n_simulation = N_SIMULATION

        if N_TRACES == 'from_input_data':
            self.N_TRACES = input_data['traces']
        else:
            self.N_TRACES = N_TRACES

        self.resources = sorted(list(input_data["resource"].keys()))
        # Synthetic placeholder resource wiring (Module 1 activities_at_risk, Phase 10):
        # add it to the resource list (observation/action space) once at construction
        # time; the corresponding SimPy RoleSimulator is (re-)registered every reset().
        if self.activities_at_risk and PLACEHOLDER_RESOURCE not in self.resources:
            self.resources = sorted(self.resources + [PLACEHOLDER_RESOURCE])

        # pool_override_path: replace the full artifact-driven resource pool with a
        # curated one (e.g. the zero-specialist pools built this session), the same
        # way every scratchpad resilience-testing script this session did by hand.
        # File format: {"curated_pt": {resource: {activity: processing_time}}, ...}.
        # Folded into gym_env itself so calibration/training/eval scripts don't each
        # re-implement this monkey-patch independently. Does not touch Module 1 or
        # make curated pools the default anywhere.
        self._pool_override_resource_names = None  # used again below, after self.params exists
        if pool_override_path is not None:
            with open(pool_override_path, 'r') as f:
                pool_data = json.load(f)
            curated_pt = pool_data['curated_pt']
            pool_names = sorted(r for r in curated_pt.keys() if r != PLACEHOLDER_RESOURCE)
            self.resource_processing_times = curated_pt
            self.resources = sorted(set(pool_names) | ({PLACEHOLDER_RESOURCE} if PLACEHOLDER_RESOURCE in curated_pt else set()))
            self._pool_override_resource_names = pool_names

        self.task_types = sorted(set(
            self._canonical_activity(t) for t in input_data["processing_time"].keys()
        ))

        # Define the input and output of the agent (neural network)
        # The inputs are: 1) resource availability, 2) if a resource is busy, to what task_type, 3) number of tasks for each activity
        self.input = [resource + '_availability' for resource in self.resources] + \
                     [resource + '_to_task_type' for resource in self.resources] + \
                     [task_type for task_type in self.task_types]
        if self.CALENDAR == True:
            self.input += ['day', 'hour']
        # D5/D7: shock signal — lets the agent distinguish shock vs. no-shock states.
        self.input += ['shock_active']
        # D6/D7: Lagrange multiplier u_k — exposes constraint pressure to the policy.
        self.input += ['u_k']
        # The outputs are: the assignments (resource, task_type)
        #self.output = [(resource, task_type) for task_type in self.task_types for resource in self.resources] + ['Postpone']

        path_model = './example/' + self.name_log + '/' + self.name_log
        if exists(path_model + '_diapr_meta.json'):
            self.FEATURE_ROLE = 'all_role'
        elif exists(path_model + '_dispr_meta.json'):
            self.FEATURE_ROLE = 'no_all_role'
        else:
            self.FEATURE_ROLE = None
        self.PATH_PETRINET = './example/' + self.name_log + '/' + self.name_log + '.pnml'
        PATH_PARAMETERS = input_file

        
        self.PATH_LOG = './example/' + self.name_log + '/' + self.name_log + '.xes'
        self.params = Parameters(PATH_PARAMETERS, self.N_TRACES, self.name_log, self.FEATURE_ROLE, threshold)

        # pool_override_path, continued: ROLE_CAPACITY can only be rebuilt now that
        # self.params exists. Keeps TRIGGER_TIMER's own entry untouched; every other
        # resource is dropped except the curated pool's, each with its calendar from
        # the full input file (default business-hours calendar if that resource has
        # none recorded there).
        if self._pool_override_resource_names is not None:
            default_cal = {"days": [0, 1, 2, 3, 4], "hour_min": 8, "hour_max": 18}
            new_role_capacity = {"TRIGGER_TIMER": self.params.ROLE_CAPACITY["TRIGGER_TIMER"]}
            for r in self._pool_override_resource_names:
                cal = input_data["resource"].get(r, {}).get("calendar", default_cal)
                new_role_capacity[r] = [[r], {"days": cal["days"], "hour_min": cal["hour_min"], "hour_max": cal["hour_max"]}]
            self.params.ROLE_CAPACITY = new_role_capacity

        # Eligible assignments: only (resource, activity) pairs present in the
        # Module 1 artifact (D4 min-support filter already applied there).
        self.output = self.retrieve_possible_assignments()
        print(len(self.input), len(self.output))
        # Observation space
        lows = np.array([0 for _ in range(len(self.input))])
        highs = np.array([1 for _ in range(len(self.input))])
        self.observation_space = spaces.Box(low=lows,
                                            high=highs,
                                            shape=(len(self.input),),
                                            dtype=np.float64)

        # Action space
        self.action_space = spaces.Discrete(len(self.output))

        self.processes = []

        self.nr_steps = 0
        self.nr_postpone = 0

        print(f'{self.name_log}, {self.N_TRACES}, calendar={self.CALENDAR}, postpone={self.postpone}, reward={self.reward_function}', flush=True)
        warnings.filterwarnings("ignore")


    # Reset the environment -> restart simulation

    def get_cluster_obs(self):
        """
        Step 18 (D8/Module 4 feed): compute the cluster-layer observation.

        Returns a dict mapping (activity, tier_label) -> count of currently
        available resources that belong to that tier for that activity.

        Called only by collect_behavioral_log.py after training — never during
        the training loop itself.  Uses process.get_state()['resource_available']
        as the availability source, which already excludes shock-removed resources
        (handled in process.py Step 11), so this is automatically shock-aware.
        """
        available = set(self.simulation_process.get_state()['resource_available'])
        counts = {}
        for resource, activity_tiers in self.resource_tier_map.items():
            for activity, tier in activity_tiers.items():
                key = (activity, tier)
                counts.setdefault(key, 0)
                if resource in available:
                    counts[key] += 1
        return counts

    def _current_ppi(self):
        """D9: phi(t) — mean cycle time of the most recently completed, non-overlapping
        batch of `ppi_window_n` traces (a tumbling window: batches never share a trace).
        Piecewise-constant — holds its value between batch completions, only updating
        once every `ppi_window_n` new completions. 0.0 before the first batch completes."""
        return self._current_phi

    def _canonical_activity(self, activity):
        """D-merge-fix: translate a native Petri-net activity label to Module
        1's merged label (no-op if this dataset has no activity_merge_map, or
        if the activity isn't one of the merged ones)."""
        return self.activity_merge_map.get(activity, activity)

    def retrieve_possible_assignments(self):
        # Build eligible (resource, activity) pairs from the Module 1 artifact.
        # A pair is eligible iff it survived the empirical processing-time computation
        # and the n_min support filter in empirical_preprocessing.py (D4).
        # This replaces the original XES-log scan and removes the pm4py read at init.
        possible_assignment = sorted([
            (resource, activity)
            for resource, activities in self.resource_processing_times.items()
            for activity in activities
        ])
        if self.postpone:
            return possible_assignment + ['Postpone']
        else:
            return possible_assignment

    def reset(self, seed=None, i=None):
        # D-repro: seed, if given, was previously accepted but never applied —
        # arrival timing (inter_trigger_timer.py), XOR-gateway branching
        # (event_trace.py), and shock sampling (shock.py) all draw from the
        # global random/np.random state with nothing controlling it, so two
        # "identical" runs produced different results. Seeding here (rather
        # than globally at import time) keeps training episodes free to stay
        # unseeded/varied (callers that don't pass seed get today's behavior
        # unchanged) while letting calibration code request reproducible runs
        # via reset(seed=k).
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        print('-------- Resetting environment --------')
        self.total_reward = 0
        self.reward_count = 0
        # D6: CPL tracking — reset each episode; lagrange_lambda persists.
        self.cpl_accumulated = 0.0
        self.prev_env_time = 0.0
        self.u_k = 0.0
        self.v_k = 0.0
        # D9: tumbling-batch PPI phi(t) tracking — reset each episode (fresh case
        # population). _phi_batch_buffer accumulates new completions not yet forming a
        # full batch; _current_phi is the last fully-resolved batch's mean (held
        # constant until the next batch completes); _phi_batches_completed counts how
        # many batches have resolved so far this episode (used by _warm_up() to know
        # when a real baseline exists).
        self._phi_batch_buffer = []
        self._current_phi = 0.0
        self._phi_batches_completed = 0
        self._phi_batches_completed_at_shock_onset = 0
        self.prev_ppi = 0.0
        self.env = simpy.Environment()
        self.simulation_process = SimulationProcess(self.env, self.params, self.CALENDAR)
        # Populate the duration lookup so event_trace.py can access it (D1).
        self.simulation_process.resource_processing_times = self.resource_processing_times
        # D-merge-fix: _resource_events (WIP-tracking, process.py) is built from
        # the native Petri-net activity keys at SimulationProcess construction
        # time, before this class's merged naming is known. Collapse any merged
        # pair onto a single shared entry keyed by the merged name, so
        # event_trace.py's _get_resource_event(action['task']) (which now always
        # sees merged names) finds it. No-op if activity_merge_map is empty.
        if self.activity_merge_map:
            merged_events = {}
            for native_key, ev in self.simulation_process._resource_events.items():
                merged_events[self._canonical_activity(native_key)] = ev
            self.simulation_process._resource_events = merged_events
        # Register the always-available placeholder resource for activities_at_risk
        # (Phase 10) — SimulationProcess is rebuilt fresh every reset(), so this must
        # be re-registered every episode.
        if self.activities_at_risk:
            self.simulation_process._resources[PLACEHOLDER_RESOURCE] = RoleSimulator(
                self.env, PLACEHOLDER_RESOURCE, math.inf, _ALWAYS_ON_CALENDAR
            )
        # D-single-shock: shock_config is still stored on simulation_process for
        # reference, but the recurring shock_scheduler is no longer started here —
        # see the warm-up + fire_single_shock block near the end of this method.
        # (shock_scheduler's train-mode infinite recurrence was exactly what made
        # next_decision_moment()'s search unbounded: env.peek() could never return
        # math.inf, since another shock was always scheduled next.)
        if self.shock_config is not None:
            self.simulation_process.shock_config = self.shock_config
        self.completed_traces = []
        #if i != None:

        if self.print:
            calendar = 'CALENDAR' if self.CALENDAR else 'NOT_CALENDAR'
            # utility.define_folder_output(f"output/output_{self.name_log}_C{self.CALENDAR}_T{self.threshold}_P{self.postpone}_{self.policy}")            
            # f = open(f"output/output_{self.name_log}_C{self.CALENDAR}_T{self.threshold}_P{self.postpone}_{self.policy}/simulated_log_{self.name_log}_{self.policy}_{str(i)}.csv", 'w')
            # print(f"output/output_{self.name_log}_C{self.CALENDAR}_T{self.threshold}_P{self.postpone}_{self.policy}/simulated_log_{self.name_log}_{self.policy}_{str(i)}.csv")
            utility.define_folder_output(f"output/output_{self.name_log}_C{self.CALENDAR}_T{self.threshold}_{self.policy}")            
            f = open(f"output/output_{self.name_log}_C{self.CALENDAR}_T{self.threshold}_{self.policy}/simulated_log_{self.name_log}_{self.policy}_{str(i)}.csv", 'w')
            print(f"output/output_{self.name_log}_C{self.CALENDAR}_T{self.threshold}_{self.policy}/simulated_log_{self.name_log}_{self.policy}_{str(i)}.csv")
            writer = csv.writer(f)
            writer.writerow(Buffer(writer).get_buffer_keys())
        else:
            writer = None
        net, im, fm = pm4py.read_pnml(self.PATH_PETRINET)
        interval = InterTriggerTimer(self.params, self.simulation_process, self.params.START_SIMULATION, self.N_TRACES)
        self.tokens = {}
        prev = 0
        for i in range(0, self.N_TRACES):
            prefix = Prefix()
            itime = interval.get_next_arrival(self.env, i, self.name_log, self.CALENDAR)
            prev = itime
            parallel_object = utility.ParallelObject()
            time_trace = self.env.now
            token = Token(i, net, im, self.params, self.simulation_process, prefix, 'sequential', writer, parallel_object,
                          itime, self.env, self.CALENDAR, None, _print=self.print)
            self.tokens[i] = token
            # D-demand-spike-fix: keep the process handle so a live DEMAND_SPIKE
            # can interrupt still-waiting arrivals (see SimulationProcess.pending_arrival_procs).
            self.simulation_process.pending_arrival_procs[i] = self.env.process(token.inter_trigger_time(itime))

        not_token_ready = True
        while not_token_ready:
            if len(self.simulation_process.tokens_pending) > 0:
                not_token_ready = False
            self.env.step()

        # D-single-shock: hidden FIFO warm-up, then fire exactly one shock, then
        # hand control to the agent. The agent's first-ever step() in this episode
        # coincides with the shock's onset instant — zero simulated time passes
        # between "shock starts" and "agent decides".
        self.shock_fired = None
        self.shock_onset_time = None
        self.shock_offset_time = None
        if self.shock_config is not None:
            warmup_steps = self._warm_up()
            if len(self.tokens) > 0:   # guard: nothing left to disrupt if warm-up exhausted N_TRACES
                shock_type, magnitude, duration = fire_single_shock(
                    self.env, self.shock_config, self.simulation_process.shock_state,
                    self.simulation_process,
                )
                self.shock_fired = (shock_type, magnitude, duration)
                self.shock_onset_time = float(self.env.now)
                self.shock_offset_time = self.shock_onset_time + duration
                # D9: snapshot how many phi(t) batches existed at shock onset, so
                # "recovered" (below) can require a genuinely NEW post-shock batch,
                # not just "shock's over and whatever phi(t) last happened to read
                # (possibly still the pre-shock baseline, if no batch has resolved
                # since) is <= P_star".
                self._phi_batches_completed_at_shock_onset = self._phi_batches_completed
                print(
                    f'Shock fired at t={self.shock_onset_time:.1f} after {warmup_steps} warm-up steps: '
                    f'type={shock_type} magnitude={magnitude:.3f} duration={duration:.1f} '
                    f'phi(t)={self._current_ppi():.1f}', flush=True,
                )
            else:
                print(
                    f'WARNING: warm-up consumed all {self.N_TRACES} traces before phi(t) '
                    f'stabilised — no shock fired this episode. Increase N_TRACES.', flush=True,
                )
            # Zero episode-scoped bookkeeping so it reflects the shock episode
            # (onset onward), not the hidden warm-up. _current_phi/prev_ppi are
            # deliberately left untouched — the warm-up-established phi(t)
            # baseline (the last fully-resolved pre-shock batch) is exactly what
            # should be reported as the "before" reference at shock onset.
            # _phi_batch_buffer IS cleared here, though: any pre-shock completions
            # not yet forming a full batch must not leak into the first post-shock
            # batch — otherwise phi(t)'s first post-shock reading would still be
            # partly pre-shock data, exactly the dilution this design exists to
            # avoid. This guarantees the first batch phi(t) resolves to, after the
            # shock fires, is built entirely from post-shock completions.
            self.total_reward = 0
            self.reward_count = 0
            self._phi_batch_buffer = []
            self.cpl_accumulated = 0.0
            self.prev_env_time = float(self.env.now)
            self.u_k = 0.0
            self.v_k = 0.0

        return self.get_state(), {}

    def _warm_up(self):
        """
        D-single-shock: run a hidden, FIFO-driven warm-up before the agent's
        first decision, until phi(t)'s first batch has resolved (ppi_window_n
        completed traces) — i.e. until phi(t) is a real baseline value rather
        than 0.0. Not exposed to the agent: reuses step() for the
        assignment/advance/reward machinery, but every reward/CPL side effect
        it produces is discarded and re-zeroed by the caller (reset())
        immediately after this returns.
        """
        max_warmup_steps = 20000
        steps = 0
        while self._phi_batches_completed < 1 and steps < max_warmup_steps:
            mask = self.action_masks()
            valid = [i for i, m in enumerate(mask) if m]
            if not valid:
                break
            action = valid[0]  # FIFO — first eligible action (matches the existing calibration convention)
            _, _, terminated, truncated, _ = self.step(action)
            steps += 1
            if terminated or truncated:
                break
        return steps

    #!! The algorithm uses this function to interact with the environment
    # Every step is an observation (state, action, reward) which is used for training
    # The agent takes an action based on the current state, and the environment should transition to the next state
    # Take an action at every timestep and evaluate this action in the simulator
    def step(self, action):
        if self.output[action] == 'Postpone':
            self.nr_postpone += 1
        self.nr_steps += 1
        if self.nr_steps % 20000 == 0:
            #env_state = self.simulation_process.get_state()
            print('Steps:', self.nr_steps, flush=True)
            #state_print = self.get_state()
            #print(state_print, flush=True)
            #print('State', [(k, state_print[i]) for i, k in enumerate(self.input)])
            #print('week/day', [env_state['time'].weekday(), env_state['time']])
            print('Postpone actions:', self.nr_postpone, '/20000', flush=True)
            self.nr_postpone = 0

        if self.output[action] != 'Postpone':
            token_id = None
            tokens_pending = {k: v for k, v in self.simulation_process.tokens_pending.items() if self._canonical_activity(v[0]._next_activity) == self.output[action][1]}
            if len(tokens_pending) > 1:
                #print(tokens_pending)
                token_id = min(tokens_pending.keys()) # FIFO_case
                #print(x[1][1] for x in tokens_pending.items())
                # token_id = max(tokens_pending.items(), key=lambda x: x[1][1])[0] #FIFO_activity
                #print(token_id)
            else:
                token_id = list(tokens_pending.keys())[0]
            simulation = self.tokens[token_id].simulation({'task': self.output[action][1],
                                                           'resource': self.output[action][0]})
            self.env.process(simulation)
            self.next_decision_moment(self.output[action]) #arg not used
        #elif self.check_exception_postpone(): ### exception case: all available resources and all tokens already arrived
        #    for token in self.simulation_process.tokens_pending:
        #        wait = self.simulation_process.tokens_pending[token][0].update_time_after_postpone(5)
        #        self.env.process(wait)
        #    self.handle_postpone_exception()
        else:
            self.next_decision_moment(self.output[action])

        ##### Reward at the end of an assignment
        '''
        trace_ongoing_actual = dict(self.simulation_process.get_state()['traces']['ongoing'])
        reward = 0
        for key in trace_ongoing_prev:
            if key in trace_ongoing_actual:
                reward += 1 - ((trace_ongoing_actual[key] - trace_ongoing_prev[key]) / CYCLE_TIME_MAX)
        for trace_id, cycle_time in self.simulation_process.traces['ended']:
            if trace_id not in self.completed_traces:
                reward += 1 - ((cycle_time - trace_ongoing_prev[trace_id]) / CYCLE_TIME_MAX)
                self.completed_traces.append(trace_id)
        '''
        #print(len(self.simulation_process.tokens_pending), len(self.simulation_process.traces['ongoing']), len(self.simulation_process.traces['ended']))
        reward = 0
        # Gather rewards
        for trace_id, cycle_time in self.simulation_process.traces['ended']:
            if trace_id not in self.completed_traces:
                self.completed_traces.append(trace_id)
                # D9: feed the tumbling-batch PPI phi(t) — shared by RF1 and RF2,
                # regardless of which reward function is active. One trace completion
                # is appended per iteration of this loop (never more than one at a
                # time), so checking "did this just complete a batch" with a plain
                # `if` right after each append is sufficient — the buffer can only
                # grow by exactly 1 between checks, so it can never overshoot
                # ppi_window_n before being caught and reset here.
                self._phi_batch_buffer.append(cycle_time)
                if len(self._phi_batch_buffer) >= self.ppi_window_n:
                    self._current_phi = float(np.mean(self._phi_batch_buffer))
                    self._phi_batch_buffer = []
                    self._phi_batches_completed += 1
                if self.reward_function == 'negative_CT':
                    reward -= cycle_time/self.normalization_cycle_time
                elif self.reward_function in ('inverse_CT', 'RF2'):
                    reward += 1 / (1 + (cycle_time/self.normalization_cycle_time))
                elif self.reward_function == 'RF1':
                    # D9 (RF1): potential-based recovery shaping + normalized deviation penalty.
                    if self.phi_fix is None or self.P_star is None:
                        raise ValueError(
                            "RF1 requires both P_star (from the Module 1 artifact) and "
                            "phi_fix (from random-policy calibration) to be set."
                        )
                    denom = abs(self.P_star - self.phi_fix) or 1e-6
                    phi_t = self._current_ppi()
                    r1 = (self.prev_ppi - phi_t) / denom          # velocity (shaping)
                    r2 = min(0.0, -(phi_t - self.P_star) / denom)  # position (deviation)
                    reward += self.beta * r1 + (1 - self.beta) * r2
                    self.prev_ppi = phi_t
                self.reward_count += 1

        # D6/D9 (RF2): SMDP CPL budget accumulation (B_k) + constrained-MDP violation
        # penalty (v_k). d(t) = max(0, phi(t) - P*) is already ~0 whenever performance
        # is at/below P*, so this naturally keeps accruing through the post-shock
        # recovery tail without needing an explicit shock-active gate.
        if self.reward_function == 'RF2':
            if self.P_star is None:
                raise ValueError("RF2 requires P_star (from the Module 1 artifact) to be set.")
            current_time = float(self.env.now)
            dt = current_time - self.prev_env_time
            if dt > 0:
                phi_t = self._current_ppi()
                d_t = max(0.0, phi_t - self.P_star)
                self.cpl_accumulated += d_t * dt          # B_k
            self.prev_env_time = current_time
            self.u_k = self.cpl_accumulated / self.cpl_budget if self.cpl_budget > 0 else 0.0
            self.v_k = max(0.0, self.u_k - 1.0)
            # f2 = 1 unconditionally: there is no postpone/no-op action, so every
            # decision step is "the agent taking an action".
            reward -= self.lagrange_lambda * self.v_k

        self.total_reward += reward

        # D-single-shock: recovery-based termination. Only applies when a shock was
        # actually fired this episode (self.shock_fired is not None) — shock-free
        # runs (shock_config=None) and the hidden warm-up loop itself are unaffected,
        # since shock_fired stays None for them. "Recovered" = the shock has ended
        # (shock_state.is_active just became False, i.e. the offset event already
        # fired) AND at least one phi(t) batch has resolved SINCE the shock fired
        # (not just "shock's over and whatever phi(t) last happened to read is
        # <= P_star" -- with a small batch size, the shock's own duration can
        # easily elapse before a single post-shock batch completes, which would
        # otherwise let a stale PRE-shock reading trigger a false "recovered")
        # AND phi(t) has dropped back to/below P*.
        recovered = (
            self.shock_fired is not None
            and self.P_star is not None
            and not self.simulation_process.shock_state.is_active
            and self._phi_batches_completed > self._phi_batches_completed_at_shock_onset
            and self._current_ppi() <= self.P_star
        )
        # No recovery backstop: an episode runs until phi(t) genuinely drops back
        # to <= P_star or the trace population is exhausted, however long that
        # takes — no time-based truncation. isTruncated is kept in the return
        # tuple only for Gymnasium API compatibility; it is never set True here.
        episode_done = len(self.tokens) == 0 or recovered
        isTruncated = False

        if episode_done:
            isTerminated = episode_done
            self.cycle_times = [cycle_time for (trace_id, cycle_time) in self.simulation_process.traces['ended']]
            self.mean_cycle_time = np.mean([cycle_time for (trace_id, cycle_time) in self.simulation_process.traces['ended']])
            if self.reward_function in ('inverse_CT', 'RF2'):
                self.total_reward_end = sum([1 / (1 + (cycle_time/self.normalization_cycle_time)) for (trace_id, cycle_time) in self.simulation_process.traces['ended']])
            elif self.reward_function == 'negative_CT':
                self.total_reward_end = sum([(-cycle_time/self.normalization_cycle_time) for (trace_id, cycle_time) in self.simulation_process.traces['ended']])
            elif self.reward_function == 'RF1':
                # RF1's reward isn't a pure post-hoc function of final cycle_times (it
                # depends on the phi(t) history built up incrementally), so there's no
                # equivalent independent "final calc" cross-check to recompute here —
                # just mirror the already-accumulated total.
                self.total_reward_end = self.total_reward
            # D6/D9 (RF2): update Lagrange multiplier at episode end, driven by the
            # violation signal v_k (spec), not the raw cpl_accumulated-cpl_budget diff.
            if self.reward_function == 'RF2':
                self.lagrange_lambda = max(0.0, self.lagrange_lambda + self.eta_lambda * self.v_k)
            print('Mean cycle time:', self.mean_cycle_time, flush=True)
            print('Total reward sum final calc (normalized):', self.total_reward_end, flush=True)
            print('Total reward cumulative:', self.total_reward, flush=True)
            print('Mean reward per traces', self.total_reward/len(self.completed_traces), flush=True)
            print('Number of completed traces', len(self.completed_traces), flush=True)
            print('Nonzero reward count', self.reward_count, flush=True)
            if recovered:
                print(f'Recovered: phi(t)={self._current_ppi():.1f} <= P_star={self.P_star:.1f}', flush=True)
        else:
            isTerminated = False
        return self.get_state(), reward, isTerminated, isTruncated, {}

    def step_baseline(self, action):
        if action is not None:
            simulation = self.tokens[action[0]].simulation({'task': action[1], 'resource': action[2]})
            self.env.process(simulation)
        self.next_decision_moment(action)

        reward = 0
        # Gather rewards
        for trace_id, cycle_time in self.simulation_process.traces['ended']:
            if trace_id not in self.completed_traces:
                self.completed_traces.append(trace_id)
                reward += 1 / (1 + (cycle_time / self.normalization_cycle_time))
        if len(self.tokens) == 0:
            isTerminated = True
            print(self.env.now)
            print('Mean cycle time:', np.mean([cycle_time for (trace_id, cycle_time) in self.simulation_process.traces['ended']]))
            self.cycle_times = [cycle_time for (trace_id, cycle_time) in self.simulation_process.traces['ended']]
            #print('gym_env',[cycle_time for (trace_id, cycle_time) in self.simulation_process.traces['ended']])
            #print([(cycle_time) for (trace_id, cycle_time) in self.simulation_process.traces['ended']])
        else:
            isTerminated = False
        return self.get_state(), reward, isTerminated, {}, {}


    def check_possible_assignments(self, resources_a, tokens):
        possible_tasks = []
        for action in self.output:
            if action[0] in resources_a:
                possible_tasks.append(action[1])
        possible_ass = False
        for token in tokens:
            if self._canonical_activity(tokens[token][0]._next_activity) in possible_tasks:
                possible_ass = True
        return possible_ass

    def delete_tokens_ended(self):
        delete = []
        for i in self.tokens:  ### update tokens ended
            if self.tokens[i].END is True:
                if self.tokens[i]._next_activity is None:
                    delete.append(i)
        for e in delete:
            del self.tokens[e]

    def check_exception_postpone(self):
        actual_state = self.simulation_process.get_state()
        state_tokens = len(actual_state['traces']['ongoing'])
        ### exception case: all available resources and all tokens already arrived
        if (state_tokens + len(actual_state['traces']['ended'])) == self.params.TRACES and len(actual_state['resource_anvailable']) == 0:
            return True
        else:
            return False

    def handle_postpone_exception(self):
        pre_cycle_time = self.simulation_process.get_state()['traces']['ongoing'].copy()
        next_step = True
        while next_step:
            if self.env.peek() == math.inf:
                next_step = False
            else:
                self.env.step()
            actual_state = self.simulation_process.get_state()['traces']['ongoing']
            if not bool(set(pre_cycle_time) & set(actual_state)):
                next_step = False

    def next_decision_moment(self, action):
        next_step = True
        pre_state_res = set(self.simulation_process.get_state()['resource_available'])
        pre_state_tokens = len(self.simulation_process.get_state()['traces']['ongoing'])
        while next_step:
            if self.env.peek() == math.inf:
                next_step = False
            else:
                self.env.step()
            actual_state = self.simulation_process.get_state()
            state_res = set(actual_state['resource_available'])
            state_tokens = len(actual_state['traces']['ongoing'])
            if pre_state_res != state_res or state_tokens > pre_state_tokens: ### state changed
                if self.check_possible_assignments(state_res, self.simulation_process.tokens_pending):
                    next_step = False
            pre_state_res = state_res
            pre_state_tokens = state_tokens
            self.delete_tokens_ended()

    def get_state(self):
        env_state = self.simulation_process.get_state()
        #if DEBUG_PRINT: print(self.env.now, self.simulation_process.get_state())
        # for each resource: 1=free, 0=occupied
        resource_available = [1.0 if resource in env_state['resource_available'] else 0 for resource in self.resources]
        # for each resource it assigns a rational number bewteen 0 and 1 with step 1/#task_types
        # every number corresponds to the task type to which the resource is assigned
        # it is like a non convetional nominal encoding done with a single real number
        resource_assigned_to = [0.0 for _ in range(len(self.resources))]
        for (trace_id, task_type, res) in env_state['actual_assignment']:
            resource_assigned_to[self.resources.index(res)] = (self.task_types.index(task_type) + 1) / len(self.task_types)
        # for each activity: it counts how many tokens_pending are waiting the resource with that activity
        # It counts until 10 and normalize by 10
        if len(self.simulation_process.tokens_pending) > 0:
            task_types_num = [min(1.0, sum([1 if self._canonical_activity(self.simulation_process.tokens_pending[token][0]._next_activity) == task_type else 0 for token in self.simulation_process.tokens_pending])/10) for task_type in self.task_types]
        else:
            task_types_num = [0.0 for _ in range(len(self.task_types))]
        #wip = [(len(env_state['traces']['ongoing'])+1)/1000]
        # D5/D7: shock signal (1.0 = shock active, 0.0 = normal).
        shock_active = [float(self.simulation_process.shock_state.is_active)]
        # D6/D9: budget-utilization ratio u_k = B_k / CPL_max (NOT lambda_k — this was
        # previously mislabeled as lambda_k here). Clamped to [0, 1] to match the
        # declared observation_space Box bounds (violation still visible via v_k/reward).
        u_k = [min(1.0, getattr(self, 'u_k', 0.0))]
        if self.CALENDAR:
            time = [env_state['time'].weekday()/6, (env_state['time'].hour*3600 + env_state['time'].minute*60 + env_state['time'].second)/(24*3600)]
            return np.array(resource_available + resource_assigned_to + task_types_num + time + shock_active + u_k)
        else:
            return np.array(resource_available + resource_assigned_to + task_types_num + shock_active + u_k)

    # Create an action mask which invalidates ineligible actions
    def action_masks(self) -> List[bool]:
        state = self.get_state()
        state_simulator = self.simulation_process.get_state()
        mask = [0 for _ in range(len(self.output))]
        for resource in self.resources:
            for task_type in self.task_types:
                if state[self.input.index(resource + '_availability')] > 0 and state[self.input.index(task_type)] > 0:
                    if (resource, task_type) in self.output:
                        mask[self.output.index((resource, task_type))] = 1
        if self.postpone:
            if len(self.simulation_process.tokens_pending) == (self.N_TRACES-len(state_simulator['traces']['ended'])) and all([state[self.input.index(resource + '_availability')] > 0 for resource in self.resources]):
                mask[-1] = 0 # All tokens have arrived and all resources available. State will not change so we mask postpone
            else:
                mask[-1] = 1 # Postpone available
        return list(map(bool, mask))

