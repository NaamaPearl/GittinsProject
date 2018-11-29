import numpy as np
import random
import queue as Q
from Prioritizer import Prioritizer, GittinsPrioritizer
from MDPModel import *
from functools import reduce
from framework import SimulatorInput, PrioritizedObject


class StateActionPair:
    P_hat_mat = []
    Q_hat_mat = []
    r_hat_mat = []

    def __init__(self, state, action):
        self.state: SimulatedState = state
        self.action = action
        self.visitations = 0
        self.TD_error = 0

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


class TDStateActionPair(StateActionPair):
    def __init__(self, state, action):
        super().__init__(state, action)

    def __gt__(self, other):
        return self.TD_error > other.TD_error


class StateActionFactory:
    @staticmethod
    def Generate(state, action, param):
        if param == 'reward':
            return StateActionPair(state, action)
        else:
            return TDStateActionPair(state, action)


class SimulatedState:
    """Represents a state with a list of possible actions from current state"""
    V_hat_vec = []
    action_num = 0
    policy = []

    def __init__(self, idx, param):
        self.idx: int = idx
        state_action_factory = StateActionFactory
        self.actions = [state_action_factory.Generate(self, action, param) for action in range(SimulatedState.action_num)]

    @property
    def V_hat(self):
        return SimulatedState.V_hat_vec[self.idx]

    @V_hat.setter
    def V_hat(self, new_val):
        SimulatedState.V_hat_vec[self.idx] = new_val

    @property
    def policy_action(self):
        return SimulatedState.policy[self.idx]

    @policy_action.setter
    def policy_action(self, new_val):
        SimulatedState.policy[self.idx] = new_val

    @property
    def s_a_visits(self):
        return self.actions[self.policy_action].visitations

    @property
    def visitations(self):
        """ sums visitation counter for all related state-action pairs"""
        return reduce(lambda a, b: a + b, [state_action.visitations for state_action in self.actions])

    @property
    def best_action(self):
        return max(self.actions)

    @property
    def r_hat(self):
        return self.actions[self.policy_action].r_hat

class Agent:
    def __init__(self, idx, init_state: SimulatedState):
        self.idx = idx
        self.curr_state: SimulatedState = init_state

    def __lt__(self, other):
        return random.choice([True, False])


class Simulator:
    def __init__(self, sim_input: SimulatorInput):
        self.MDP_model: MDPModel = sim_input.MDP_model
        state_num = self.MDP_model.n
        self.gamma = sim_input.gamma
        self.epsilon = sim_input.epsilon

        self.r_hat = np.zeros((state_num, self.MDP_model.actions))
        self.P_hat = [np.zeros((self.MDP_model.actions, state_num)) for _ in range(state_num)]
        self.policy = [random.randint(0, self.MDP_model.actions - 1) for _ in range(state_num)]
        self.V_hat = np.zeros(state_num)
        self.Q_hat = np.zeros((state_num, self.MDP_model.actions))

        self.InitStatics()

        self.states = [SimulatedState(idx, sim_input.parameter) for idx in range(state_num)]
        self.graded_states = {state.idx: random.random() for state in self.states}
        self.init_prob = self.MDP_model.init_prob
        self.agents = None
        self.reset_agents(sim_input.agent_num)

    def InitStatics(self):
        SimulatedState.policy = self.policy
        SimulatedState.action_num = self.MDP_model.actions
        SimulatedState.V_hat_vec = self.V_hat

        StateActionPair.P_hat_mat = self.P_hat
        StateActionPair.Q_hat_mat = self.Q_hat
        StateActionPair.r_hat_mat = self.r_hat

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

    def ApproxModel(self, prioritizer: Prioritizer):
        #self.graded_states = prioritizer.GradeStates(self.states, self.policy, self.P_hat, self.r_hat)
        self.graded_states = prioritizer.GradeStates(self.states, self.policy, self.MDP_model.P, self.MDP_model.r)
        self.ImprovePolicy()
        self.ReGradeAllAgents()

    def ReGradeAllAgents(self):
        """invoked after states re-prioritization. Replaces queue"""
        new_queue = Q.PriorityQueue()
        while self.agents.qsize() > 0:
            new_queue.put(self.GradeAgent(self.agents.get().object))

        self.agents = new_queue

    def GradeAgent(self, agent):
        return PrioritizedObject(agent, self.graded_states[agent.curr_state.idx])

    def simulate(self, prioritizer, steps=10000, grades_freq=100, reset_freq=1000):
        self.InitStatics()

        for i in range(steps):
            self.SimulateOneStep()
            if i % grades_freq == grades_freq - 1:
                self.ApproxModel(prioritizer)  # prioritize agents & states

            if i % reset_freq == reset_freq - 1:
                self.reset_agents(self.agents.qsize())

            if (i % 5000) == 0:
                print('simulate step %d' % i)

    def SimulateOneStep(self, agents_to_run=1):
        """find top-priority agents, and activate them for a single step"""
        agents_list = (self.agents.get().object for _ in range(agents_to_run))
        for agent in agents_list:
            self.SimulateAgent(agent)
            self.agents.put(self.GradeAgent(agent))

    def ChooseAction(self, state: SimulatedState):
        if random.random() < self.epsilon:
            return np.random.choice(state.actions)
        else:
            return state.actions[state.policy_action]

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
        new_val = (state_action.r_hat * state_action.visitations + new_reward) / (state_action.visitations + 1)
        state_action.r_hat = new_val

    def evaluate_P_hat(self):
        return self.P_hat_sum_diff()

    def calculate_V(self):
        return np.linalg.inv(np.eye(self.MDP_model.n) - self.gamma * self.curr_policy_P) @ self.curr_policy_reward

    @property
    def curr_policy_P(self):
        return np.array([self.MDP_model.P[i][a] for (i, a) in enumerate(self.policy)])

    @property
    def curr_policy_reward(self):
        return np.array([self.MDP_model.r[i][a] for (i, a) in enumerate(self.policy)])

    def reset_agents(self, agents_num):
        self.agents = Q.PriorityQueue()
        [self.agents.put(PrioritizedObject(Agent(i, self.ChooseInitState()))) for i in range(agents_num)]




if __name__ == '__main__':
    n = 4
    actions=2
    k = 1

    MDP = SingleLineMDP(n=n, actions=actions)
    simulator_input = SimulatorInput(MDP)
    random_simulator = Simulator(simulator_input)
    # gittins_simulator = Simulator(simulator_input)

    #   random_simulator.simulate(Prioritizer(), steps=100000)
    random_simulator.simulate(GittinsPrioritizer(), steps=10000)

    print('r difference')
    print('------------')
    print(np.array(random_simulator.r_hat - random_simulator.MDP_model.r))

    print('P difference')
    print('------------')
    for s in range(random_simulator.MDP_model.n):
        print(np.array(random_simulator.P_hat[s] - random_simulator.MDP_model.P[s]))

    print('Policy')
    print('------------')
    print(random_simulator.policy)

    print('V difference')
    print('------------')
    V_diff = random_simulator.calculate_V() - random_simulator.V_hat
    print(V_diff)
    print('percentage of error')
    print(abs(V_diff / random_simulator.calculate_V()) * 100)

    print('Q Function')
    print(random_simulator.Q_hat)

    print('visits')
    print([[random_simulator.states[idx].actions[a].visitations for a in range(MDP.actions)] for idx in range(MDP.n)])

    print('all done')
