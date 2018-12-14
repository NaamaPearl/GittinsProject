import queue as Q
from Prioritizer import Prioritizer, GittinsPrioritizer
from MDPModel import *
from functools import reduce
from framework import *
from Critic import *
import heapq


class StateActionPair:
    P_hat_mat = []
    Q_hat_mat = []
    TD_error_mat = []
    r_hat_mat = []
    T_bored_num = 3
    T_bored_val = 5

    def __init__(self, state, action):
        self.state: SimulatedState = state
        self.action = action
        self.visitations = 0

    def __gt__(self, other):
        assert isinstance(other, StateActionPair)
        return self.Q_hat > other.Q_hat

    @property
    def r_hat(self):
        return StateActionPair.r_hat_mat[self.state.idx][self.action]

    @r_hat.setter
    def r_hat(self, new_val):
        StateActionPair.r_hat_mat[self.state.idx][self.action] = new_val

    @property
    def P_hat(self):
        return StateActionPair.P_hat_mat[self.state.idx][self.action]

    @P_hat.setter
    def P_hat(self, new_val):
        StateActionPair.P_hat_mat[self.state.idx][self.action] = new_val

    def UpdateVisits(self):
        self.visitations += 1

    @property
    def Q_hat(self):
        return StateActionPair.Q_hat_mat[self.state.idx][self.action]

    @Q_hat.setter
    def Q_hat(self, new_val):
        StateActionPair.Q_hat_mat[self.state.idx][self.action] = new_val

    @property
    def TD_error(self):
        if self.visitations < StateActionPair.T_bored_num:
            return StateActionPair.T_bored_val + abs(StateActionPair.TD_error_mat[self.state.idx][self.action])
        return StateActionPair.TD_error_mat[self.state.idx][self.action]

    @TD_error.setter
    def TD_error(self, new_val):
        StateActionPair.TD_error_mat[self.state.idx][self.action] = new_val


class SimulatedState:
    """Represents a state with a list of possible actions from current state"""
    V_hat_vec = []
    action_num = 0
    policy = []

    def __init__(self, idx, chain):
        self.idx: int = idx
        self.actions = [StateActionPair(self, a) for a in range(SimulatedState.action_num)]
        self.chain = chain

    @property
    def V_hat(self):
        return SimulatedState.V_hat_vec[self.idx]

    @V_hat.setter
    def V_hat(self, new_val):
        SimulatedState.V_hat_vec[self.idx] = new_val

    @property
    def policy_action(self):
        return self.actions[SimulatedState.policy[self.idx]]

    @policy_action.setter
    def policy_action(self, new_val):
        SimulatedState.policy[self.idx] = new_val

    @property
    def s_a_visits(self):
        return self.policy_action.visitations

    @property
    def visitations(self):
        """ sums visitation counter for all related state-action pairs"""
        return reduce(lambda a, b: a + b, [state_action.visitations for state_action in self.actions])

    @property
    def best_action(self):
        return max(self.actions)

    @property
    def r_hat(self):
        return self.policy_action.r_hat


class Agent:
    def __init__(self, idx, init_state: SimulatedState):
        self.idx = idx
        self.curr_state: SimulatedState = init_state

    def __lt__(self, other):
        return random.choice([True, False])

    @property
    def chain(self):
        return self.curr_state.chain


class SimulatedModel:
    def __init__(self, mdp_model):
        self.MDP_model: MDPModel = mdp_model
        self.policy_dynamics = np.zeros((mdp_model.n, mdp_model.n))
        self.policy_expected_rewards = np.zeros(mdp_model.n)

    def CalcPolicyData(self, policy):
        for i, a in enumerate(policy):
            self.policy_dynamics[i] = self.MDP_model.P[i][a]
            self.policy_expected_rewards[i] = self.MDP_model.r[i][a][0]

    def GetNextState(self, state_action, run_time=1):
        n_s = np.random.choice(range(self.MDP_model.n), p=self.MDP_model.P[state_action.state.idx][state_action.action])
        if run_time == 1:
            return n_s

        p = self.policy_dynamics ** (run_time - 1)
        return np.random.choice(range(self.MDP_model.n), p=p[n_s])

    def calculate_V(self, gamma):
        return np.linalg.inv(np.eye(self.MDP_model.n) - gamma * self.policy_dynamics) @ self.policy_expected_rewards

    def GetReward(self, state_action):
        params = self.MDP_model.r[state_action.state.idx][state_action.action]
        if self.MDP_model.reward_type == 'gauss':
            return np.random.normal(params[0], params[1])
        if self.MDP_model.reward_type == 'bernuly':
            return params[0] * np.random.binomial(1, params[1])

        # if time_to_run > 1:
        #     position_vec = np.zeros(self.MDP_model.n)
        #     position_vec[state_action.state.idx] = 1
        #
        #     for i in range(time_to_run - 1):  # first simulation is made in the previous row
        #         position_vec = self.policy_dynamics @ position_vec
        #         reward += (gamma ** i * (position_vec @ self.policy_expected_rewards))
        #
        # return reward

    @property
    def n(self):
        return self.MDP_model.n

    @property
    def actions(self):
        return self.MDP_model.actions

    @property
    def init_prob(self):
        return self.MDP_model.init_prob

    def FindChain(self, idx):
        return self.MDP_model.FindChain(idx)

    @property
    def chain_num(self):
        return self.MDP_model.chain_num

    @property
    def init_states_idx(self):
        return self.MDP_model.init_states_idx


class Simulator:
    def __init__(self, sim_input: ProblemInput):
        self.MDP_model: SimulatedModel = sim_input.MDP_model
        self.gamma = sim_input.gamma
        self.epsilon = sim_input.epsilon
        self.evaluated_model = EvaluatedModel()
        self.evaluate_policy = None
        self.policy = None
        self.critic = None
        self.states = None

        self.InitParams()
        self.InitStatics()

    def InitStatics(self):
        SimulatedState.policy = self.policy
        SimulatedState.V_hat_vec = self.evaluated_model.V_hat

        StateActionPair.P_hat_mat = self.evaluated_model.P_hat
        StateActionPair.Q_hat_mat = self.evaluated_model.Q_hat
        StateActionPair.TD_error_mat = self.evaluated_model.TD_error
        StateActionPair.r_hat_mat = self.evaluated_model.r_hat

    def InitParams(self):
        self.critic = CriticFactory.Generate(type=self.MDP_model.MDP_model.type, chain_num=self.MDP_model.chain_num)
        state_num = self.MDP_model.n
        SimulatedState.action_num = self.MDP_model.actions
        self.evaluate_policy = []
        self.evaluated_model.ResetData(self.MDP_model.n, self.MDP_model.actions)
        self.states = [SimulatedState(idx, self.FindChain(idx)) for idx in range(state_num)]
        self.policy = [random.randint(0, self.MDP_model.actions - 1) for _ in range(state_num)]
        self.MDP_model.CalcPolicyData(self.policy)

    def FindChain(self, idx):
        return self.MDP_model.FindChain(idx)

    def ImprovePolicy(self, *argv):
        for state in self.states:
            state.policy_action = state.best_action.action

        self.MDP_model.CalcPolicyData(self.policy)

    def Update_V(self, state_action: StateActionPair):
        #  self.V_hat[idx] = self.r_hat[idx] + self.gamma * np.dot(self.P_hat[action][idx, :], self.V_hat)
        state = state_action.state
        state.V_hat = state_action.r_hat + self.gamma * state_action.P_hat @ self.evaluated_model.V_hat

    def Update_Q(self, state_action: StateActionPair, next_state: SimulatedState, reward: int):
        a_n = (state_action.visitations + 1) ** -0.7
        state_action.TD_error = reward + self.gamma * max(next_state.actions).Q_hat - state_action.Q_hat
        state_action.Q_hat += a_n * state_action.TD_error

    def SampleStateAction(self, state_action: StateActionPair):
        next_state = self.states[self.MDP_model.GetNextState(state_action)]
        reward = self.MDP_model.GetReward(state_action)
        self.UpdateReward(state_action, reward)
        self.UpdateP(state_action, next_state)

        self.Update_V(state_action)
        self.Update_Q(state_action, next_state, reward)
        state_action.UpdateVisits()

        return next_state

    @staticmethod
    def UpdateP(state_action: StateActionPair, next_state: SimulatedState):
        curr_num_of_tran = state_action.P_hat * state_action.visitations
        curr_num_of_tran[next_state.idx] += 1

        new_est_p_row = curr_num_of_tran / (state_action.visitations + 1)
        state_action.P_hat = new_est_p_row

    @staticmethod
    def UpdateReward(state_action: StateActionPair, new_reward: int):
        state_action.r_hat = (state_action.r_hat * state_action.visitations + new_reward) / (
                state_action.visitations + 1)

    def EvaluatePolicy(self, trajectory_len):
        reward = 0
        good_agents = 0
        for _ in range(50):
            agent = Agent(0, self.states[np.random.choice(list(self.MDP_model.init_states_idx))])
            agent.curr_state = self.states[self.MDP_model.GetNextState(agent.curr_state.policy_action)]
            if agent.curr_state.chain == 0:
                continue

            good_agents += 1
            for _ in range(trajectory_len):
                reward += self.MDP_model.GetReward(agent.curr_state.policy_action)
                agent.curr_state = self.states[self.MDP_model.GetNextState(agent.curr_state.policy_action)]

        self.evaluate_policy.append(reward / good_agents)

    def simulate(self, sim_input):
        self.InitParams()
        self.InitStatics()

        for i in range(sim_input.steps):
            self.SimulateOneStep(agents_to_run=sim_input.agents_to_run)
            if i % sim_input.grades_freq == sim_input.grades_freq - 1:
                self.ImprovePolicy(sim_input, i)

            if i % sim_input.reset_freq == sim_input.reset_freq - 1:
                self.EvaluatePolicy(50)

    def SimulateOneStep(self, agents_to_run):
        pass

    # @property
    # def V_hat(self):
    #     return self.evaluated_model.V_hat
    #
    # @property
    # def Q_hat(self):
    #     return self.evaluated_model.Q_hat
    #
    # @property
    # def P_hat(self):
    #     return self.evaluated_model.P_hat
    #
    # @property
    # def r_hat(self):
    #     return self.evaluated_model.r_hat
    #
    # @property
    # def TD_error(self):
    #     return self.evaluated_model.TD_error
    #
    # @property
    # def policy(self):
    #     return self.evaluated_model.policy


class AgentSimulator(Simulator):
    def __init__(self, sim_input: ProblemInput):
        self.agents_num = sim_input.agent_num
        super().__init__(sim_input)
        self.agents = Q.PriorityQueue()
        self.graded_states = None
        self.init_prob = None
        self.agents_to_run = None

    def InitParams(self):
        super().InitParams()
        self.graded_states = {state.idx: random.random() for state in self.states}
        self.init_prob = self.MDP_model.init_prob
        self.ResetAgents(self.agents_num)
        self.graded_states = {state.idx: random.random() for state in self.states}
        self.init_prob = self.MDP_model.init_prob

    def ChooseInitState(self):
        return np.random.choice(self.states, p=self.init_prob)

    def GetStatsForGittins(self, parameter):
        if parameter is None:
            return
        if parameter == 'reward':
            return self.evaluated_model.P_hat, self.evaluated_model.r_hat
        if parameter == 'error':
            return self.evaluated_model.P_hat, abs(self.evaluated_model.TD_error)

        reward_mat = [[self.MDP_model.MDP_model.r[state][action][0] * self.MDP_model.MDP_model.r[state][action][1]
                       for action in range(self.MDP_model.actions)] for state in range(self.MDP_model.n)]
        return self.MDP_model.MDP_model.P, reward_mat

    def ImprovePolicy(self, *argv):
        sim_input: AgentSimulationInput = argv[0]
        iteration_num = argv[1]
        self.graded_states = sim_input.prioritizer.GradeStates(self.states,
                                                               self.policy,
                                                               self.GetStatsForGittins(sim_input.parameter),
                                                               iteration_num < self.MDP_model.n * 4)
        self.ReGradeAllAgents()
        super().ImprovePolicy()

    def EvaluatePolicy(self, trajectory_len):
        self.ResetAgents(self.agents.qsize())
        super().EvaluatePolicy(trajectory_len)

    def ReGradeAllAgents(self):
        """invoked after states re-prioritization. Replaces queue"""
        new_queue = Q.PriorityQueue()
        while self.agents.qsize() > 0:
            agent = self.agents.get().object
            new_queue.put(self.GradeAgent(agent))

        self.agents = new_queue

    def GradeAgent(self, agent):
        return PrioritizedObject(agent, self.graded_states[agent.curr_state.idx])

    def SimulateOneStep(self, agents_to_run):
        """find top-priority agents, and activate them for a single step"""
        agents_list = [self.agents.get().object for _ in range(agents_to_run)]
        for agent in agents_list:
            self.SimulateAgent(agent)
            self.critic.Update(agent.chain)
            self.agents.put(self.GradeAgent(agent))

    def ChooseAction(self, state: SimulatedState):
        if random.random() < self.epsilon or state.visitations < (self.MDP_model.actions * 5):
            return np.random.choice(state.actions)
        return state.policy_action

    def SimulateAgent(self, agent: Agent):
        """simulate one action of an agent, and re-grade it, according to it's new state"""
        state_action = self.ChooseAction(agent.curr_state)

        next_state = self.SampleStateAction(state_action)
        agent.curr_state = next_state

    def ResetAgents(self, agents_num):
        self.agents = Q.PriorityQueue()
        for i in range(agents_num):
            init_state = self.ChooseInitState()
            self.agents.put(PrioritizedObject(Agent(i, init_state), -np.inf))

    @property
    def agents_location(self):
        return [agent.object.curr_state.idx for agent in self.agents.queue]


class PrioritizedSweeping(Simulator):
    def __init__(self, sim_input: ProblemInput):
        super().__init__(sim_input)
        state_actions_list = reduce(lambda a, b: a+b, [state.actions for state in self.states])
        self.state_action = [PrioritizedObject(state_action, -np.inf) for state_action in state_actions_list]
        heapq.heapify(self.state_action)

    def SimulateOneStep(self, agents_to_run):
        for _ in range(agents_to_run):
            state_action: StateActionPair = heapq.heappop(self.state_action).object
            next_state = self.SampleStateAction(state_action)
            self.critic.Update(next_state.chain)

            score = -np.inf if state_action.visitations < StateActionPair.T_bored_num else -abs(state_action.TD_error)
            heapq.heappush(self.state_action, PrioritizedObject(state_action, score))


# class ChainsSimulator(Simulator):
#     def __init__(self, sim_input: SimulatorInput):
#         self.chain_activations = None
#         super().__init__(sim_input)
#
#     def UpdateActivations(self):
#         res = np.zeros(self.MDP_model.chain_num)
#         for state in set(self.graded_states).difference(self.MDP_model.init_states_idx):
#             res[self.MDP_model.FindChain(state)] += (self.MDP_model.n - self.graded_states[state])
#
#         for i in range(self.MDP_model.chain_num):
#             self.chain_activations[i].append(res[i])
#
#     # def PlotVisitations(self, mat, title):
#     #     plt.figure()
#     #     visitation_sum = np.cumsum(mat, axis=1)
#     #     [plt.plot(visitation_sum[i]) for i in range(chains_input.MDP_model.actions)]
#     #     plt.legend(['chain ' + str(c) for c in range(chains_input.MDP_model.actions)])
#     #     plt.title(title)
#     #     plt.show()


def SimulatorFactory(method_type, mdp, agents_to_run):
    if method_type == 'random':
        return AgentSimulator(ProblemInput(SimulatedModel(mdp), agent_num=agents_to_run))
    if method_type == 'reward':
        return AgentSimulator(ProblemInput(SimulatedModel(mdp), agent_num=agents_to_run * 3))
    if method_type == 'error':
        return AgentSimulator(ProblemInput(SimulatedModel(mdp), agent_num=agents_to_run * 3))
    if method_type == 'sweeping':
        return PrioritizedSweeping(ProblemInput(SimulatedModel(mdp), agent_num=1))
    raise IOError('unrecognized methid type:' + method_type)


def SimInputFactory(method_type, simulation_steps, agents_to_run):
    if method_type == 'random':
        return AgentSimulationInput(prioritizer=Prioritizer(), steps=simulation_steps, parameter=None,
                                    agents_to_run=agents_to_run)
    if method_type == 'reward':
        return AgentSimulationInput(prioritizer=GittinsPrioritizer(), steps=simulation_steps,
                                    parameter='reward',
                                    agents_to_run=agents_to_run)
    if method_type == 'error':
        return AgentSimulationInput(prioritizer=GittinsPrioritizer(), steps=simulation_steps,
                                    parameter='error',
                                    agents_to_run=agents_to_run)
    if method_type == 'sweeping':
        return SimulationInput(steps=simulation_steps, agents_to_run=agents_to_run)

    raise IOError('unrecognized method type:' + method_type)
