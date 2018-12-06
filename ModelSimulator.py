import queue as Q
from Prioritizer import Prioritizer, GittinsPrioritizer
from MDPModel import *
from functools import reduce
from framework import SimulatorInput, PrioritizedObject
from matplotlib import pyplot as plt


class StateActionPair:
    P_hat_mat = []
    Q_hat_mat = []
    TD_error_mat = []
    r_hat_mat = []
    T_bored_num = 1
    T_bored_val = 0

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
        if self.visitations < StateActionPair.T_bored_num:
            return StateActionPair.T_bored_val
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

    def __init__(self, idx):
        self.idx: int = idx
        self.actions = [StateActionPair(self, action) for action in range(SimulatedState.action_num)]

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


class Simulator:
    def __init__(self, sim_input: SimulatorInput):
        self.MDP_model: MDPModel = sim_input.MDP_model
        self.agents_num = sim_input.agent_num
        self.gamma = sim_input.gamma
        self.epsilon = sim_input.epsilon

        self.r_hat = None
        self.P_hat = None
        self.policy = None
        self.V_hat = None
        self.Q_hat = None
        self.TD_error = None
        self.agents = Q.PriorityQueue()
        self.states = None
        self.graded_states = None
        self.init_prob = None

        self.InitParams()
        self.InitStatics()

        self.tmp_reward = []
        self.fork_action = []
        self.param = sim_input.parameter

    def InitStatics(self):
        SimulatedState.policy = self.policy
        SimulatedState.action_num = self.MDP_model.actions
        SimulatedState.V_hat_vec = self.V_hat

        StateActionPair.P_hat_mat = self.P_hat
        StateActionPair.Q_hat_mat = self.Q_hat
        StateActionPair.TD_error_mat = self.TD_error
        StateActionPair.r_hat_mat = self.r_hat

    def InitParams(self):
        state_num = self.MDP_model.n
        self.r_hat = np.zeros((state_num, self.MDP_model.actions))
        self.P_hat = [np.zeros((self.MDP_model.actions, state_num)) for _ in range(state_num)]
        self.policy = [random.randint(0, self.MDP_model.actions - 1) for _ in range(state_num)]
        self.V_hat = np.zeros(state_num)
        self.Q_hat = np.zeros((state_num, self.MDP_model.actions))
        self.TD_error = np.zeros((state_num, self.MDP_model.actions))

        self.states = [SimulatedState(idx) for idx in range(state_num)]
        self.graded_states = {state.idx: random.random() for state in self.states}
        self.init_prob = self.MDP_model.init_prob
        self.ResetAgents(self.agents_num)

    def ChooseInitState(self):
        return np.random.choice(self.states, p=self.init_prob)

    def ImprovePolicy(self):
        for state in self.states:
            state.policy_action = state.best_action.action

    def Update_V(self, state_action: StateActionPair):
        #  self.V_hat[idx] = self.r_hat[idx] + self.gamma * np.dot(self.P_hat[action][idx, :], self.V_hat)
        state = state_action.state
        state.V_hat = state_action.r_hat + self.gamma * state_action.P_hat @ self.V_hat

    def Update_Q(self, state_action: StateActionPair, next_state: SimulatedState, reward: int):
        a_n = (state_action.visitations + 1) ** -0.7  # TODO: state-action visits or state visits?
        state_action.TD_error = reward + self.gamma * max(next_state.actions).Q_hat - state_action.Q_hat
        state_action.Q_hat += a_n * state_action.TD_error

    def P_hat_sum_diff(self):
        return [abs(self.MDP_model.P[a] - self.P_hat[a]).mean() for a in range(self.MDP_model.actions)]

    # def V_hat_diff(self):
    #     res = []
    #     for action in range(self.MDP_model.actions):
    #         V = np.dot(self.MDP_model.r,
    #                    np.linalg.inv(np.eye(self.MDP_model.n) - self.gamma * self.MDP_model.P[action]))
    #         print(V)
    #         res.append(abs(self.V_hat - V).max())

    def GetStatsForGittins(self):
        if self.param == 'reward':
            return self.r_hat
        return abs(self.TD_error)

    def ApproxModel(self, prioritizer: Prioritizer):
        self.graded_states = prioritizer.GradeStates(self.states, self.policy, self.P_hat, self.GetStatsForGittins())
        self.ImprovePolicy()
        self.ReGradeAllAgents()

    def ReGradeAllAgents(self):
        """invoked after states re-prioritization. Replaces queue"""
        new_queue = Q.PriorityQueue()
        while self.agents.qsize() > 0:
            agent = self.agents.get().object
            new_queue.put(self.GradeAgent(agent))

        self.agents = new_queue

    def GradeAgent(self, agent):
        return PrioritizedObject(agent, self.graded_states[agent.curr_state.idx])

    def simulate(self, prioritizer, _steps, grades_freq=10, reset_freq=20):
        self.InitParams()
        self.InitStatics()

        for i in range(_steps):
            self.SimulateOneStep(agents_to_run=3)
            if i % grades_freq == grades_freq - 1:
                self.ApproxModel(prioritizer)  # prioritize agents & states

            if i % reset_freq == reset_freq - 1:
                self.ResetAgents(self.agents.qsize())

            if (i % 500) == 0:
                print('simulate step %d' % i)

    def CalcRegret(self):
        wrong = 0
        for location in self.agents_location:
            if location in self.MDP_model.chains[1]:
                wrong += 1

        return wrong / self.agents.qsize()

    def SimulateOneStep(self, agents_to_run=1):
        """find top-priority agents, and activate them for a single step"""
        agents_list = [self.agents.get().object for _ in range(agents_to_run)]
        self.UpdateActivations(agents_list)
        for agent in agents_list:
            self.SimulateAgent(agent)
            self.agents.put(self.GradeAgent(agent))

    def UpdateActivations(self, agent_list):
        pass

    def ChooseAction(self, state: SimulatedState):
        if random.random() < self.epsilon or state.visitations < 5:
            return np.random.choice(state.actions)
        return state.policy_action

    def SimulateAgent(self, agent: Agent):
        """simulate one action of an agent, and re-grade it, according to it's new state"""
        state_action = self.ChooseAction(agent.curr_state)

        next_state = np.random.choice(self.states, p=self.MDP_model.P[state_action.state.idx][state_action.action])
        reward = self.MDP_model.GetReward(state_action.state, state_action.action)
        self.UpdateReward(state_action, reward)
        self.UpdateP(state_action, next_state)

        self.Update_V(state_action)
        self.Update_Q(state_action, next_state, reward)
        state_action.UpdateVisits()
        agent.curr_state = next_state

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

    def evaluate_P_hat(self):
        return self.P_hat_sum_diff()

    def calculate_V(self):
        return np.linalg.inv(np.eye(self.MDP_model.n) - self.gamma * self.curr_policy_P) @ self.curr_policy_reward

    @property
    def curr_policy_reward(self):
        return np.array([self.MDP_model.r[state][a][0] for (state, a) in enumerate(self.policy)])

    @property
    def curr_policy_P(self):
        return np.array([self.MDP_model.P[i][a] for (i, a) in enumerate(self.policy)])

    def ResetAgents(self, agents_num):
        self.agents = Q.PriorityQueue()
        for i in range(agents_num):
            init_state = self.ChooseInitState()
            self.agents.put(PrioritizedObject(Agent(i, init_state), -np.inf))

    def PrintAgentsStatus(self):
        for i, agent in enumerate(self.agents.queue):
            print("Agent #%s is in state #%s" % (i, agent.object.curr_state.idx))

    @property
    def agents_location(self):
        return [agent.object.curr_state.idx for agent in self.agents.queue]


class ChainsSimulator(Simulator):
    def __init__(self, sim_input: SimulatorInput):
        self.chain_visits = [[] for _ in range(sim_input.MDP_model.chain_num)]
        self.chain_activations = [[] for _ in range(sim_input.MDP_model.chain_num)]
        super().__init__(sim_input)

    @property
    def curr_chain_visits(self):
        res = np.zeros(self.MDP_model.chain_num)
        for state_idx in self.agents_location:
            res[self.MDP_model.FindChain(state_idx)] += 1

        return res

    def ResetAgents(self, agents_num):
        tmp = self.curr_chain_visits
        for i in range(self.MDP_model.chain_num):
            self.chain_visits[i].append(tmp[i])

        super().ResetAgents(agents_num)

    def UpdateActivations(self, agent_list):
        res = np.zeros(self.MDP_model.chain_num)
        for agent in agent_list:
            agents_chain = self.MDP_model.FindChain(agent.curr_state.idx)
            if agents_chain is None:
                continue
            res[agents_chain] += 1

        for i in range(self.MDP_model.chain_num):
            self.chain_activations[i].append(res[i])


if __name__ == '__main__':
    steps = 10000
    chains_input = SimulatorInput(SeperateChainsMDP(n=9), 'error')
    chains_simulator = ChainsSimulator(chains_input)
    chains_simulator.simulate(Prioritizer(), _steps=steps)

    plt.figure()
    visitation_sum = np.cumsum(chains_simulator.chain_visits, axis=1)
    [plt.plot(visitation_sum[i]) for i in range(chains_input.MDP_model.actions)]
    plt.legend(['chain ' + str(c) for c in range(chains_input.MDP_model.actions)])
    plt.title('Accumulated Visitations for all chains')
    plt.show()
    # print('r difference')
    # print('------------')
    # print(np.array(random_simulator.r_hat - random_simulator.MDP_model.r[0]))
    #
    # # print('P difference')
    # # print('------------')
    # # for s in range(random_simulator.MDP_model.n):
    # #     print(np.array(random_simulator.P_hat[s] - random_simulator.MDP_model.P[s]))
    # #
    # # print('Policy')
    # # print('------------')
    # # print(random_simulator.policy)
    #
    # print('V difference')
    # print('------------')
    # V_diff = random_simulator.calculate_V() - random_simulator.V_hat
    # print(V_diff)
    # print('percentage of error')
    # print(abs(V_diff / random_simulator.calculate_V()) * 100)

    # print('Q Function')
    # print(random_simulator.Q_hat)
    #
    # print('visits')
    # print([[random_simulator.states[idx].actions[a].visitations for a in range(MDP.actions)] for idx in range(MDP.n)])

    print('all done')
