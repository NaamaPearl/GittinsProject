import heapq
import queue as Q
from collections import Counter

from Actor.Prioritizer import *
from Critic.Critic import *
from Framework.Inputs import *
from Framework.PrioritizedObject import PrioritizedObject
from Simulator.SimulatorBasics import *


class Simulator:
    """ Abstract class which tries to learn optimal policy via Q-Learning, based on observations """

    def __init__(self, sim_input: ProblemInput):
        self.MDP_model: SimulatedModel = sim_input.MDP_model
        self.evaluation_type = sim_input.eval_type
        self.gamma = sim_input.gamma
        self.epsilon = sim_input.epsilon
        self.evaluated_model = EvaluatedModel()

        self.critic = CriticFactory.Generate(model=self.MDP_model, evaluator_type=self.evaluation_type)
        state_num = self.MDP_model.n
        SimulatedState.action_num = self.MDP_model.actions
        self.evaluated_model.ResetData(self.MDP_model.n, self.MDP_model.actions)
        self.policy = [0] * state_num
        self.MDP_model.CalcPolicyData(self.policy)

        # Initiate static variables
        SimulatedState.policy = self.policy
        SimulatedState.V_hat_vec = self.evaluated_model.V_hat

        StateActionPair.P_hat_mat = self.evaluated_model.P_hat
        StateActionPair.Q_hat_mat = self.evaluated_model.Q_hat
        StateActionPair.TD_error_mat = self.evaluated_model.TD_error
        StateActionPair.r_hat_mat = self.evaluated_model.r_hat
        StateActionPair.visitations_mat = self.evaluated_model.visitations

    def ImprovePolicy(self, sim_input, **kwargs):
        """ Choose best action per state, based on Q value"""
        for state in self.MDP_model.states:
            state.policy_action = state.best_action.action

        self.MDP_model.CalcPolicyData(self.policy)

    def getActionResults(self, state_action: StateActionPair):
        """ simulates desired action, and returns next_state, reward """
        next_state, reward = self.MDP_model.MDP_model.sample_state_action(state_action.state.idx, state_action.action)
        return self.MDP_model.states[next_state], reward

    def updateModel(self, current_state_action, next_state, reward):
        def Update_V():
            state = current_state_action.state
            future_v = current_state_action.P_hat @ self.evaluated_model.V_hat
            state.V_hat = current_state_action.r_hat + self.gamma * future_v

        def Update_Q():
            a_n = (current_state_action.visitations + 1) ** -0.7
            current_state_action.TD_error = reward + self.gamma * max(
                next_state.actions).Q_hat - current_state_action.Q_hat
            current_state_action.Q_hat += (a_n * current_state_action.TD_error)

        def UpdateP():
            curr_num_of_tran = current_state_action.P_hat * current_state_action.visitations
            curr_num_of_tran[next_state.idx] += 1

            new_est_p_row = curr_num_of_tran / (current_state_action.visitations + 1)
            current_state_action.P_hat = new_est_p_row

        def UpdateReward():
            current_state_action.r_hat = (current_state_action.r_hat * current_state_action.visitations + reward) / (
                    current_state_action.visitations + 1)

        UpdateReward()
        UpdateP()
        Update_V()
        Update_Q()
        current_state_action.UpdateVisits()

    def SampleStateAction(self, agent_type, state_action: StateActionPair):
        next_state, reward = self.getActionResults(state_action)
        if agent_type == 'regular':
            self.updateModel(state_action, next_state, reward)

        return reward, next_state

    def simulate(self, sim_input):
        self.init_simulation(sim_input)
        for i in range(int(sim_input.steps / sim_input.temporal_extension)):
            self.SimulateOneStep(agents_to_run=sim_input.agents_to_run,
                                 temporal_extension=sim_input.temporal_extension,
                                 iteration_num=i,
                                 T_board=sim_input.T_board)
            if i % sim_input.grades_freq == 0:
                self.ImprovePolicy(sim_input, iteration_num=i)
            if i % sim_input.evaluate_freq == 0:  # sim_input.evaluate_freq - 1:
                self.SimEvaluate(trajectory_len=sim_input.trajectory_len, running_agents=sim_input.agents_to_run,
                                 gamma=self.gamma)
            # if i % sim_input.reset_freq == 0:  # sim_input.reset_freq - 1:
            #     self.Reset()

        return self.critic

    def Reset(self):
        pass

    def init_simulation(self, sim_input):
        pass

    @abstractmethod
    def SimulateOneStep(self, agents_to_run, **kwargs):
        pass

    def SimEvaluate(self, **kwargs):
        self.critic.CriticEvaluate(initial_state=self.RaffleInitialState(), good_agents=50,
                                   chain_num=self.MDP_model.MDP_model.chain_num,
                                   active_chains_ratio=self.MDP_model.MDP_model.active_chains_ratio,
                                   active_chains=self.MDP_model.MDP_model.GetActiveChains(),
                                   **kwargs)

    def RaffleInitialState(self):
        return np.random.choice(self.MDP_model.states, p=self.MDP_model.MDP_model.init_prob)

    @property
    def opt_policy(self):
        return self.MDP_model.MDP_model.opt_policy


class AgentSimulator(Simulator):
    def __init__(self, sim_input: ProblemInput):
        super().__init__(sim_input)
        self.agents_num = sim_input.agent_num
        self.init_prob = self.MDP_model.init_prob
        self.agents = Q.PriorityQueue()
        self.optimal_agents = self.generateOptimalAgents(sim_input.agent_num)
        self.ResetAgents(self.agents_num)

        self.graded_states = {state.idx: (state.idx, random.random()) for state in self.MDP_model.states}

    def generateOptimalAgents(self, agents_num):
        agents_list = []
        good_agents = 0
        while good_agents < agents_num:
            new_agent = Agent(100 + good_agents, self.RaffleInitialState(), agent_type='optimal')
            next_state, _ = self.getActionResults(self.ChooseAction(new_agent.curr_state, new_agent.type))
            if next_state.chain not in self.MDP_model.MDP_model.GetActiveChains():
                continue
            new_agent.curr_state = next_state
            agents_list.append(new_agent)
            good_agents += 1

        return agents_list

    def SimEvaluate(self, **kwargs):
        kwargs['agents_reward'] = [agent.object.getOnlineAndZero() for agent in self.agents.queue]
        kwargs['optimal_agents_reward'] = [agent.getOnlineAndZero() for agent in self.optimal_agents]
        super().SimEvaluate(**kwargs)

    def ChooseInitState(self):
        return np.random.choice(self.MDP_model.states, p=self.init_prob)

    def GetStatsForPrioritizer(self, parameter):
        if parameter is None:
            return None, None
        if parameter == 'reward':
            return self.evaluated_model.P_hat, self.evaluated_model.r_hat
        if parameter == 'error':
            return self.evaluated_model.P_hat, abs(self.evaluated_model.TD_error)
        if parameter == 'ground_truth':
            return self.MDP_model.MDP_model.P, np.transpose(self.MDP_model.MDP_model.expected_r)
        if parameter == 'model_free':
            return self.MDP_model.MDP_model, None

    def ImprovePolicy(self, sim_input, **kwargs):
        """
        :param sim_input: simulation parameters
        :param kwargs: must contain current iteration number, for reincarnation
        :effect: calculate new indexes for all sates, and grade agents accordingly
        """
        super().ImprovePolicy(sim_input)

        p, r = self.GetStatsForPrioritizer(sim_input.parameter)
        prioritizer = sim_input.prioritizer(states=self.MDP_model.states,
                                            policy=self.policy,
                                            p=p,
                                            r=r,
                                            temporal_extension=sim_input.temporal_extension,
                                            discount_factor=sim_input.gittins_discount,
                                            trajectory_num=sim_input.trajectory_num,
                                            max_trajectory_len=sim_input.max_trajectory_len)
        self.graded_states = prioritizer.GradeStates()

        self.ReGradeAllAgents(kwargs['iteration_num'], sim_input.grades_freq)

    def ReincarnateAgent(self, agent, iteration_num, grades_freq):
        pass
        # if iteration_num - agent.last_activation > 10000 * grades_freq:
        #     agent.last_activation = iteration_num
        #     agent.curr_state = self.RaffleInitialState()

    def ReGradeAllAgents(self, iteration_num, grades_freq):
        """invoked after states re-prioritization. Replaces queue"""
        new_queue = Q.PriorityQueue()
        while self.agents.qsize() > 0:
            agent = self.agents.get().object
            self.ReincarnateAgent(agent, iteration_num, grades_freq)
            new_queue.put(self.GradeAgent(agent))

        self.agents = new_queue

    def GradeAgent(self, agent):
        """ Agents in non-visited states / initial states are prioritized"""
        state = agent.curr_state.idx
        score = (0, -np.inf) if state in self.MDP_model.init_states_idx else self.graded_states[state]
        return PrioritizedObject(agent, score)

    def SimulateOneStep(self, agents_to_run, **kwargs):
        """ Find top-priority agents, and activate them for a single step"""
        possible_states = [agent.object.curr_state.idx for agent in self.agents.queue]

        agents_list = [self.agents.get().object for _ in range(agents_to_run)]
        activated_states = [agent.curr_state.idx for agent in agents_list]

        self.optimal_agents = self.optimal_agents[:agents_to_run]

        for agent in agents_list + self.optimal_agents:
            for _ in range(kwargs['temporal_extension']):
                self.SimulateAgent(agent, **kwargs)
                if agent.type == 'regular':
                    self.critic.Update(agent.chain, agent.curr_state.idx)

            if agent.type == 'regular':
                self.agents.put(self.GradeAgent(agent))

        return possible_states, activated_states

    def ChooseAction(self, state: SimulatedState, agent_type, T_board=0):
        if agent_type == 'optimal':
            return state.actions[self.opt_policy[state.idx]]

        if agent_type == 'regular':
            min_visits, min_action = state.min_visitations
            if min_visits < T_board:
                return state.actions[min_action]

            return state.policy_action if random.random() > self.epsilon else np.random.choice(state.actions)

    def SimulateAgent(self, agent: Agent, iteration_num, **kwargs):
        """simulate one action of an agent, and re-grade it, according to it's new state"""

        state_action = self.ChooseAction(agent.curr_state, agent.type, kwargs['T_board'])

        agent.last_activation = iteration_num
        reward, next_state = self.SampleStateAction(agent.type, state_action)
        agent.update(reward, next_state)

    def Reset(self):
        self.ResetAgents(self.agents.qsize())

    def ResetAgents(self, agents_num):
        self.agents = Q.PriorityQueue()
        for i in range(agents_num):
            init_state = self.ChooseInitState()
            self.agents.put(PrioritizedObject(Agent(i, init_state), (-np.inf, 0)))

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

    def SimEvaluate(self, **kwargs):
        kwargs['bad_activated_states'] = self.bad_activated_states
        super().SimEvaluate(**kwargs)

    def SimulateOneStep(self, agents_to_run, **kwargs):
        possible_states, activated_states = super().SimulateOneStep(agents_to_run, **kwargs)

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
        self.indexes_vec.append([self.graded_states[key][1] for key in range(self.MDP_model.MDP_model.n)])

        p_gt, r_gt = self.GetStatsForPrioritizer('ground_truth')
        gt_prioritizer = GittinsPrioritizer(states=self.MDP_model.states,
                                            policy=self.policy,
                                            p=p_gt,
                                            r=r_gt,
                                            temporal_extension=sim_input.temporal_extension,
                                            discount_factor=sim_input.gittins_discount)

        self.gittins = gt_prioritizer.GradeStates()
        self.gt_indexes_vec.append([self.gittins[key][1] for key in range(self.MDP_model.MDP_model.n)])

    def init_simulation(self, sim_input):
        self.calc_index_vec(sim_input)

    def ImprovePolicy(self, sim_input, **kwargs):
        super().ImprovePolicy(sim_input, **kwargs)
        self.calc_index_vec(sim_input)


def SimulatorFactory(mdp: MDPModel, sim_params, gt_compare):
    if gt_compare:
        simulator = GTAgentSimulator
    else:
        simulator = AgentSimulator

    return simulator(
        ProblemInput(MDP_model=SimulatedModel(mdp), gamma=mdp.gamma, **sim_params))


def SimInputFactory(method_type, parameter, sim_params):
    simulation_input_type = AgentSimulationInput

    if method_type == 'random':
        parameter = None
        prioritizer = Prioritizer
    elif method_type == 'gittins':
        prioritizer = ModelFreeGittinsPrioritizer if parameter == 'model_free' else GittinsPrioritizer
    elif method_type == 'greedy':
        prioritizer = GreedyPrioritizer
    else:
        raise IOError('unrecognized method type:' + method_type)

    return simulation_input_type(prioritizer=prioritizer, parameter=parameter, **sim_params)
