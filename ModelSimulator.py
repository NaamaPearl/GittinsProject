import numpy as np
import random
import queue as Q
from Prioritizer import Prioritizer, GittinsPrioritizer
from MDPModel import MDPModel
from functools import reduce
from framework import SimulatorInput, PrioritizedObject


class StateActionPair:
    P_hat_mat = []
    Q_hat_mat = []

    def __init__(self, state, action):
        self.state: SimulatedState = state
        self.action = action
        self.visitations = 0
        self.P_hat = np.zeros(len(self.P_hat_mat[self.state.idx][self.action]))

    def __gt__(self, other):
        return self.Q_hat > other.Q_hat

    @property
    def state_reward(self):
        return self.state.r_hat

    @property
    def state_V(self):
        return self.state.V_hat

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


class SimulatedState:
    """
    Represents a state with a list of possible actions from current state
    """
    r_hat_vec = []
    V_hat_vec = []
    action_num = 0
    policy = []

    def __init__(self, idx):
        self.idx = idx
        self.actions = [StateActionPair(self, action) for action in range(SimulatedState.action_num)]

    @property
    def V_hat(self):
        return SimulatedState.V_hat_vec[self.idx]

    @V_hat.setter
    def V_hat(self, new_val):
        SimulatedState.V_hat_vec[self.idx] = new_val

    @property
    def policy_action(self):
        return SimulatedState.policy[self.idx]

    @property
    def r_hat(self):
        return SimulatedState.r_hat_vec[self.idx]

    @r_hat.setter
    def r_hat(self, new_val):
        SimulatedState.r_hat_vec[self.idx] = new_val

    @property
    def visitations(self):
        """ sums visitation counter for all related state-action pairs"""
        return reduce(lambda a, b: a + b, [state_action.visitations for state_action in self.actions])

    def ImprovePolicy(self):
        """Finds action with maximal approximated Q value, and updates policy"""
        SimulatedState.policy[self.idx] = max(self.actions).action


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

        self.r_hat = np.zeros(state_num)
        SimulatedState.r_hat_vec = self.r_hat

        self.P_hat = [np.zeros((self.MDP_model.actions, state_num)) for _ in range(state_num)]
        StateActionPair.P_hat_mat = self.P_hat

        self.policy = [random.randint(0, self.MDP_model.actions - 1) for _ in range(state_num)]
        SimulatedState.policy = self.policy
        SimulatedState.action_num = self.MDP_model.actions

        self.gamma = sim_input.gamma
        self.V_hat = np.zeros(state_num)
        SimulatedState.V_hat_vec = self.V_hat

        self.Q_hat = np.zeros((state_num, self.MDP_model.actions))
        StateActionPair.Q_hat_mat = self.Q_hat
        self.states = [SimulatedState(idx) for idx in range(state_num)]
        self.graded_states = {state.idx: random.random() for state in self.states}
        self.agents = Q.PriorityQueue()
        self.init_prob = sim_input.init_prob
        [self.agents.put(PrioritizedObject(Agent(i, self.ChooseInitState()), i)) for i in range(sim_input.agent_num)]

    def ChooseInitState(self):
        return np.random.choice(self.states, p=self.init_prob)

    def ImprovePolicy(self):
        for state in self.states:
            state.ImprovePolicy()

    def Update_V(self, state_action: StateActionPair):
        #  self.V_hat[idx] = self.r_hat[idx] + self.gamma * np.dot(self.P_hat[action][idx, :], self.V_hat)
        state = state_action.state
        state.V_hat = state.r_hat + self.gamma * state_action.P_hat @ self.V_hat

    def Update_Q(self, state_action: StateActionPair):
        state_action.Q_hat = state_action.state_reward + self.gamma * state_action.P_hat @ self.V_hat

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
        self.graded_states = prioritizer.GradeStates(self.states, self.policy, self.P_hat, self.r_hat)
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

    def simulate(self, prioritizer, steps=10000, grades_freq=20):
        for i in range(steps):
            self.SimulateOneStep()
            if i % grades_freq == grades_freq - 1:
                self.ApproxModel(prioritizer)  # prioritize agents & states

    # find top-priority agents, and activate them for a single step
    def SimulateOneStep(self, agents_to_run=1):
        agents_list = (self.agents.get().object for _ in range(agents_to_run))
        for agent in agents_list:
            self.SimulateAgent(agent)

    def SimulateAgent(self, agent: Agent):
        """simulate one action of an agent, and re-grade it, according to it's new state"""
        action = agent.curr_state.policy_action
        state_action = agent.curr_state.actions[action]

        next_state = np.random.choice(self.states, p=self.MDP_model.P[agent.curr_state.idx][action])
        reward = self.MDP_model.GetReward(state_action.state)
        self.UpdateReward(agent.curr_state, next_state, reward)
        self.UpdateP(state_action, next_state)

        self.Update_V(state_action)
        self.Update_Q(state_action)
        state_action.UpdateVisits()
        agent.curr_state = next_state

        self.agents.put(self.GradeAgent(agent))

    @staticmethod
    def UpdateP(state_action: StateActionPair, next_state: SimulatedState):
        curr_num_of_tran = state_action.P_hat * state_action.visitations
        curr_num_of_tran[next_state.idx] += 1

        new_est_p_row = curr_num_of_tran / (state_action.visitations + 1)
        state_action.P_hat = new_est_p_row

    @staticmethod
    def UpdateReward(curr_state: SimulatedState, next_state: SimulatedState, new_reward):
        new_val = (curr_state.r_hat * next_state.visitations + new_reward) / (next_state.visitations + 1)
        curr_state.r_hat = new_val

    def evaluate_P_hat(self):
        return self.P_hat_sum_diff()

    # def EvaluateV(self):
    #     return self.V_hat_diff()


if __name__ == '__main__':
    n = 10
    k = 4

    MDP = MDPModel(n=n)
    simulator_input = SimulatorInput(MDP)
    #   random_simulator = Simulator(MDP)
    gittins_simulator = Simulator(simulator_input)

    #   random_simulator.simulate(Prioritizer(), steps=100000)
    gittins_simulator.simulate(GittinsPrioritizer(), steps=10000)

    print('eval Random')
    #   print(random_simulator.evaluate_P_hat())
    #   print(random_simulator.EvaluateV())

    print('eval Gittin')
    print(gittins_simulator.evaluate_P_hat())
    #   print(gittins_simulator.EvaluateV())

    # print(GittinsSimulator.evaluateGittins())
    print('all done')
