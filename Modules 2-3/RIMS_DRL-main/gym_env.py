import gymnasium as gym
from gymnasium import spaces, Env
import numpy as np
from typing import List
import csv
import simpy
import utility
from process import SimulationProcess
from event_trace import Token
from parameters import Parameters
import sys, getopt
from utility import *
import pm4py
from inter_trigger_timer import InterTriggerTimer
import warnings
import json
import math
import pm4py
from os.path import exists
from datetime import datetime, timedelta
import random
CYCLE_TIME_MAX = 8.64e+6


DEBUG_PRINT = False
if __name__ == "__main__":
    warnings.filterwarnings("ignore")


class gym_env(Env):
    def __init__(self, NAME_LOG, N_TRACES, CALENDAR, POLICY=None, N_SIMULATION=0, threshold=0, postpone=True, reward_function='inverse_CT', artifact_path=None, shock_config=None, cpl_budget=0.0, eta_lambda=0.01) -> None:
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

        self.shock_config = shock_config  # D5: ShockConfig instance or None
        # D6 (RF2): constrained-MDP parameters.
        self.cpl_budget = cpl_budget       # target max cumulative process lag per episode
        self.eta_lambda = eta_lambda       # Lagrange multiplier step size
        self.lagrange_lambda = 0.0         # u_k — persists across episodes, updated at episode end

        self.n_simulation = N_SIMULATION

        if N_TRACES == 'from_input_data':
            self.N_TRACES = input_data['traces']
        else:
            self.N_TRACES = N_TRACES

        self.resources = sorted(list(input_data["resource"].keys()))
        self.task_types = sorted(list(input_data["processing_time"].keys()))

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

    def reset(self, seed=0, i=None):
        print('-------- Resetting environment --------')
        self.total_reward = 0
        self.reward_count = 0
        # D6: CPL tracking — reset each episode; lagrange_lambda persists.
        self.cpl_accumulated = 0.0
        self.prev_env_time = 0.0
        self.env = simpy.Environment()
        self.simulation_process = SimulationProcess(self.env, self.params, self.CALENDAR)
        # Populate the duration lookup so event_trace.py can access it (D1).
        self.simulation_process.resource_processing_times = self.resource_processing_times
        # D5: register the shock SimPy process for this episode if shocks are configured.
        if self.shock_config is not None:
            self.simulation_process.shock_config = self.shock_config
            self.simulation_process.start_shock_scheduler(self.env)
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
            self.env.process(token.inter_trigger_time(itime))

        not_token_ready = True
        while not_token_ready:
            if len(self.simulation_process.tokens_pending) > 0:
                not_token_ready = False
            self.env.step()
        return self.get_state(), {}

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
            tokens_pending = {k: v for k, v in self.simulation_process.tokens_pending.items() if v[0]._next_activity == self.output[action][1]}
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
                if self.reward_function == 'negative_CT':
                    reward -= cycle_time/self.normalization_cycle_time
                elif self.reward_function in ('inverse_CT', 'RF2'):
                    reward += 1 / (1 + (cycle_time/self.normalization_cycle_time))
                self.reward_count += 1

        # D6 (RF2): accumulate CPL and apply Lagrange penalty per step.
        if self.reward_function == 'RF2':
            current_time = float(self.env.now)
            dt = current_time - self.prev_env_time
            if dt > 0 and self.simulation_process.shock_state.is_active:
                n_pending = len(self.simulation_process.tokens_pending)
                cpl_delta = dt * n_pending / self.normalization_cycle_time
                self.cpl_accumulated += cpl_delta
                reward -= self.lagrange_lambda * cpl_delta
            self.prev_env_time = current_time

        self.total_reward += reward

        if len(self.tokens) == 0:
            isTerminated = True
            self.cycle_times = [cycle_time for (trace_id, cycle_time) in self.simulation_process.traces['ended']]
            self.mean_cycle_time = np.mean([cycle_time for (trace_id, cycle_time) in self.simulation_process.traces['ended']])
            if self.reward_function in ('inverse_CT', 'RF2'):
                self.total_reward_end = sum([1 / (1 + (cycle_time/self.normalization_cycle_time)) for (trace_id, cycle_time) in self.simulation_process.traces['ended']])
            elif self.reward_function == 'negative_CT':
                self.total_reward_end = sum([(-cycle_time/self.normalization_cycle_time) for (trace_id, cycle_time) in self.simulation_process.traces['ended']])
            # D6 (RF2): update Lagrange multiplier at episode end.
            if self.reward_function == 'RF2':
                self.lagrange_lambda = max(0.0, self.lagrange_lambda + self.eta_lambda * (self.cpl_accumulated - self.cpl_budget))
            print('Mean cycle time:', self.mean_cycle_time, flush=True)
            print('Total reward sum final calc (normalized):', self.total_reward_end, flush=True)
            print('Total reward cumulative:', self.total_reward, flush=True)
            print('Mean reward per traces', self.total_reward/len(self.completed_traces), flush=True)
            print('Number of completed traces', len(self.completed_traces), flush=True)
            print('Nonzero reward count', self.reward_count, flush=True)
        else:
            isTerminated = False
        return self.get_state(), reward, isTerminated, {}, {}

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
            if tokens[token][0]._next_activity in possible_tasks:
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
            task_types_num = [min(1.0, sum([1 if self.simulation_process.tokens_pending[token][0]._next_activity == task_type else 0 for token in self.simulation_process.tokens_pending])/10) for task_type in self.task_types]
        else:
            task_types_num = [0.0 for _ in range(len(self.task_types))]
        #wip = [(len(env_state['traces']['ongoing'])+1)/1000]
        # D5/D7: shock signal (1.0 = shock active, 0.0 = normal).
        shock_active = [float(self.simulation_process.shock_state.is_active)]
        # D6/D7: Lagrange multiplier u_k (clamped to [0, 1] for observation stability).
        u_k = [min(1.0, self.lagrange_lambda)]
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

