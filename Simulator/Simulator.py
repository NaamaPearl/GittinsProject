import heapq
import queue as Q
import numpy as np
from collections import Counter
from abc import abstractmethod

import random

import Actor.Prioritizer as Pr
from Critic.Critic import CriticFactory
from Framework.Inputs import ProblemInput, AgentSimulationInput
from Framework.PrioritizedObject import PrioritizedObject
import Simulator.SimulatorBasics as sb
import MDPModel.MDPBasics as mdp


class Simulator:
    """ Abstract class which tries to learn optimal policy via Q-Learning, based on observations """

    def __init__(self, sim_input: ProblemInput):
        self.model: sb.SimulatedModel = sim_input.MDP_model
        self.evaluation_type = sim_input.eval_type
        self.gamma = sim_input.MDP_model.MDP.gamma
        self.epsilon = sim_input.epsilon
        self.evaluated_model = mdp.EvaluatedModel()

        self.critic = CriticFactory.generate(model=self.model, evaluator_type=self.evaluation_type)
        state_num = self.model.n
        sb.SimulatedState.action_num = self.model.actions
        self.evaluated_model.ResetData(self.model.n, self.model.actions)
        self.policy = [0] * state_num
        self.model.calc_policy_data(self.policy)

        # Initiate static variables
        sb.SimulatedState.policy = self.policy
        sb.SimulatedState.V_hat_vec = self.evaluated_model.V_hat

        mdp.StateActionPair.Q_hat_mat = self.evaluated_model.Q_hat
        mdp.StateActionPair.TD_error_mat = self.evaluated_model.TD_error
        mdp.StateActionPair.r_hat_mat = self.evaluated_model.r_hat
        mdp.StateActionPair.P_hat_mat = self.evaluated_model.P_hat
        mdp.StateActionPair.visitations_mat = self.evaluated_model.visitations

    def improve_policy(self, sim_input, **kwargs):
        """ Choose best action per state, based on Q value"""
        for state in self.model.states:
            state.policy_action = state.best_action.action

        self.model.calc_policy_data(self.policy)

    def get_action_results(self, state_action: mdp.StateActionPair):
        """ simulates desired action, and returns next_state, reward """
        next_state, reward = self.model.MDP.sample_state_action(state_action.state.idx, state_action.action)
        return self.model.states[next_state], reward

    def update_model(self, current_state_action, next_state, reward):
        def update_v():
            future_v = current_state_action.P_hat @ self.evaluated_model.V_hat
            current_state_action.state.V_hat = current_state_action.r_hat + self.gamma * future_v

        def update_q():
            a_n = (current_state_action.visitations + 1) ** -0.7
            current_state_action.TD_error = reward + self.gamma * max(
                next_state.actions).Q_hat - current_state_action.Q_hat
            current_state_action.Q_hat += (a_n * current_state_action.TD_error)

        def update_p():
            curr_num_of_tran = current_state_action.P_hat * current_state_action.visitations
            curr_num_of_tran[next_state.idx] += 1

            new_est_p_row = curr_num_of_tran / (current_state_action.visitations + 1)
            current_state_action.P_hat = new_est_p_row

        def update_reward():
            current_state_action.r_hat = (current_state_action.r_hat * current_state_action.visitations + reward) / (
                    current_state_action.visitations + 1)

        update_reward()
        update_p()
        update_v()
        update_q()
        current_state_action.UpdateVisits()

    def sample_state_action(self, agent_type, state_action: mdp.StateActionPair):
        next_state, reward = self.get_action_results(state_action)
        if agent_type == 'regular':
            self.update_model(state_action, next_state, reward)

        return reward, next_state

    def simulate(self, sim_input):
        self.init_simulation(sim_input)
        for i in range(int(sim_input.steps / sim_input.temporal_extension)):
            self.simulate_one_step(agents_to_run=sim_input.agents_to_run,
                                   temporal_extension=sim_input.temporal_extension,
                                   iteration_num=i,
                                   T_board=sim_input.T_board)
            if i % sim_input.grades_freq == 0:
                self.improve_policy(sim_input, iteration_num=i)
            if i % sim_input.evaluate_freq == 0:  # sim_input.evaluate_freq - 1:
                self.sim_evaluate(trajectory_len=sim_input.trajectory_len,
                                  running_agents=sim_input.agents_to_run,
                                  gamma=self.gamma)
            # if i % sim_input.reset_freq == 0:  # sim_input.reset_freq - 1:
            #     self.Reset()

        return self.critic

    def reset(self):
        pass

    def init_simulation(self, sim_input):
        pass

    @abstractmethod
    def simulate_one_step(self, agents_to_run, **kwargs):
        pass

    def sim_evaluate(self, **kwargs):
        self.critic.critic_evaluate(initial_state=self.raffle_initial_state(), good_agents=50,
                                    chain_num=self.model.MDP.chain_num,
                                    active_chains_ratio=self.model.MDP.active_chains_ratio,
                                    active_chains=self.model.MDP.get_active_chains(),
                                    **kwargs)

    def raffle_initial_state(self):
        return np.random.choice(self.model.states, p=self.model.MDP.init_prob)

    @property
    def opt_policy(self):
        return self.model.MDP.opt_policy


class AgentSimulator(Simulator):
    def __init__(self, sim_input: ProblemInput):
        super().__init__(sim_input)
        self.agents_num = sim_input.agent_num
        self.init_prob = self.model.init_prob
        self.agents = Q.PriorityQueue()
        self.optimal_agents = self.generate_optimal_agents(sim_input.agent_num)
        self.reset_agents(self.agents_num)

        self.graded_states = {state.idx: (state.idx, random.random()) for state in self.model.states}

    def generate_optimal_agents(self, agents_num):
        agents_list = []
        good_agents = 0
        while good_agents < agents_num:
            new_agent = sb.Agent(100 + good_agents, self.raffle_initial_state(), agent_type='optimal')
            next_state, _ = self.get_action_results(self.choose_action(new_agent.curr_state, new_agent.type))
            if next_state.chain not in self.model.MDP.get_active_chains():
                continue
            new_agent.curr_state = next_state
            agents_list.append(new_agent)
            good_agents += 1

        return agents_list

    def sim_evaluate(self, **kwargs):
        kwargs['agents_reward'] = list(map(lambda agent: agent.object.get_online_and_zero(), self.agents.queue))
        kwargs['optimal_agents_reward'] = list(map(lambda agent: agent.get_online_and_zero(), self.optimal_agents))
        super().sim_evaluate(**kwargs)

    def choose_init_state(self):
        return np.random.choice(self.model.states, p=self.init_prob)

    def get_stats_for_prioritizer(self, method, parameter):
        if parameter is None:
            return None, None

        if method == 'model_free':
            return self.model.MDP, self.evaluated_model.TD_error if parameter == 'error' else None

        if parameter == 'reward':
            return self.evaluated_model.P_hat, self.evaluated_model.r_hat
        if parameter == 'error':
            return self.evaluated_model.P_hat, abs(self.evaluated_model.TD_error)
        if parameter == 'v_f':
            return self.evaluated_model.P_hat, self.evaluated_model.V_hat
        if parameter == 'ground_truth':
            return self.model.MDP.P, np.transpose(self.model.MDP.expected_r)

    def improve_policy(self, sim_input, **kwargs):
        """
        :param sim_input: simulation parameters
        :param kwargs: must contain current iteration number, for reincarnation
        :effect: calculate new indexes for all sates, and grade agents accordingly
        """
        super().improve_policy(sim_input)

        p, r = self.get_stats_for_prioritizer(sim_input.method, sim_input.parameter)
        prioritizer = sim_input.prioritizer(states=self.model.states,
                                            policy=self.policy,
                                            p=p,
                                            r=r,
                                            temporal_extension=sim_input.temporal_extension,
                                            discount_factor=sim_input.gittins_discount,
                                            trajectory_num=sim_input.trajectory_num,
                                            max_trajectory_len=sim_input.max_trajectory_len,
                                            parameter=sim_input.parameter)
        self.graded_states = prioritizer.grade_states()

        self.regrade_all_agents(kwargs['iteration_num'], sim_input.grades_freq)

    def reincarnate_agent(self, agent, iteration_num, grades_freq):
        pass
        # if iteration_num - agent.last_activation > 10000 * grades_freq:
        #     agent.last_activation = iteration_num
        #     agent.curr_state = self.RaffleInitialState()

    def regrade_all_agents(self, iteration_num, grades_freq):
        """invoked after states re-prioritization. Replaces queue"""
        new_queue = Q.PriorityQueue()
        while self.agents.qsize() > 0:
            agent = self.agents.get().object
            self.reincarnate_agent(agent, iteration_num, grades_freq)
            new_queue.put(self.grade_agent(agent))

        self.agents = new_queue

    def grade_agent(self, agent):
        """ Agents in non-visited states / initial states are prioritized"""
        state = agent.curr_state.idx
        score = (0, -np.inf) if state in self.model.init_states_idx else self.graded_states[state]
        return PrioritizedObject(agent, score)

    def simulate_one_step(self, agents_to_run, **kwargs):
        """ Find top-priority agents, and activate them for a single step"""
        possible_states = [agent.object.curr_state.idx for agent in self.agents.queue]

        agents_list = [self.agents.get().object for _ in range(agents_to_run)]
        activated_states = [agent.curr_state.idx for agent in agents_list]

        self.optimal_agents = self.optimal_agents[:agents_to_run]

        for agent in agents_list + self.optimal_agents:
            for _ in range(kwargs['temporal_extension']):
                self.simulate_agent(agent, **kwargs)
                if agent.type == 'regular':
                    self.critic.update(agent.chain, agent.curr_state.idx)

            if agent.type == 'regular':
                self.agents.put(self.grade_agent(agent))

        return possible_states, activated_states

    def choose_action(self, state: sb.SimulatedState, agent_type, t_board=0):
        if agent_type == 'optimal':
            return state.actions[self.opt_policy[state.idx]]

        if agent_type == 'regular':
            min_visits, min_action = state.min_visitations
            if min_visits < t_board:
                return state.actions[min_action]

            return state.policy_action if random.random() > self.epsilon else np.random.choice(state.actions)

    def simulate_agent(self, agent: sb.Agent, iteration_num, **kwargs):
        """simulate one action of an agent, and re-grade it, according to it's new state"""

        state_action = self.choose_action(agent.curr_state, agent.type, kwargs['T_board'])

        agent.last_activation = iteration_num
        reward, next_state = self.sample_state_action(agent.type, state_action)
        agent.update(reward, next_state)

    def reset(self):
        self.reset_agents(self.agents.qsize())

    def reset_agents(self, agents_num):
        self.agents = Q.PriorityQueue()
        for i in range(agents_num):
            init_state = self.choose_init_state()
            self.agents.put(PrioritizedObject(sb.Agent(i, init_state), (-np.inf, 0)))

    @property
    def agents_location(self):
        chains_count = Counter([agent.object.chain for agent in self.agents.queue])
        return chains_count


class GTAgentSimulator(AgentSimulator):
    def __init__(self, sim_input: ProblemInput):
        self.bad_activated_states = 0
        self.gittins = {}
        self.indexes_vec = []
        self.gt_indexes_vec = []
        super().__init__(sim_input)

    def simulate(self, sim_input):
        return super().simulate(sim_input), self.indexes_vec[1:], self.gt_indexes_vec[1:]

    def sim_evaluate(self, **kwargs):
        kwargs['bad_activated_states'] = self.bad_activated_states
        super().sim_evaluate(**kwargs)

    def simulate_one_step(self, agents_to_run, **kwargs):
        possible_states, activated_states = super().simulate_one_step(agents_to_run, **kwargs)

        wrongly_activated = 0
        states_order = [(self.gittins[state][0], state) for state in possible_states]
        heapq.heapify(states_order)
        optimal_states = [heapq.heappop(states_order)[1] for _ in range(len(activated_states))]

        optimal_counter = Counter(optimal_states)
        chosen_counter = Counter(activated_states)

        # a list of states that are optimal, but weren't activated by the prioritizer
        optimal_not_activated = [self.gittins[state][1] for state in (optimal_counter - chosen_counter).elements()]

        # iterate through states that were activated, but aren't optimal.
        # If a sub-optimal activated state's gittins index is equal to one from the optimal not activated,
        # choosing it was not a mistake
        for state in (chosen_counter - optimal_counter).elements():
            state_grade = self.gittins[state][1]
            if state_grade in optimal_not_activated:
                optimal_not_activated.remove(state_grade)
            else:
                wrongly_activated += 1

        self.bad_activated_states += wrongly_activated

    def calc_index_vec(self, sim_input):
        self.indexes_vec.append([self.graded_states[key][1] for key in range(self.model.MDP.n)])

        p_gt, r_gt = self.get_stats_for_prioritizer('gittins', sim_input.parameter)
        gt_prioritizer = Pr.GittinsPrioritizer(states=self.model.states,
                                               policy=self.policy,
                                               p=p_gt,
                                               r=r_gt,
                                               temporal_extension=sim_input.temporal_extension,
                                               discount_factor=sim_input.gittins_discount)

        self.gittins = gt_prioritizer.grade_states()
        self.gt_indexes_vec.append([self.gittins[key][1] for key in range(self.model.MDP.n)])

    def init_simulation(self, sim_input):
        self.calc_index_vec(sim_input)

    def improve_policy(self, sim_input, **kwargs):
        super().improve_policy(sim_input, **kwargs)
        self.calc_index_vec(sim_input)


def SimulatorFactory(new_mdp: sb.MDPModel, sim_params, gt_compare):
    if gt_compare:
        simulator = GTAgentSimulator
    else:
        simulator = AgentSimulator

    return simulator(ProblemInput(sim_params, sb.SimulatedModel(new_mdp)))


def SimInputFactory(method_type, parameter, sim_params):
    simulation_input_type = AgentSimulationInput

    if method_type == 'random':
        parameter = None
        prioritizer = Pr.Prioritizer
    elif method_type == 'gittins':
        prioritizer = Pr.GittinsPrioritizer
    elif method_type == 'greedy':
        prioritizer = Pr.GreedyPrioritizer
    elif method_type == 'model_free':
        prioritizer = Pr.ModelFreeGittinsPrioritizer
    else:
        raise IOError('unrecognized method type:' + method_type)

    return simulation_input_type(method_type, prioritizer, parameter, sim_params)
