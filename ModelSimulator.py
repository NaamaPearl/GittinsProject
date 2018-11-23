import numpy as np
import random
import queue as Q
from Prioritizer import Prioritizer, GittinsPrioritizer
from MDPModel import MDPModel, PrioritizedObject
from functools import reduce


class StateActionPair:
    def __init__(self, state, action, P_hat_mat):
        self.P_hat_mat = P_hat_mat
        self.state = state
        self.action = action
        self.visitations = 0
        self.P_hat = P_hat_mat[self.action][self.state.idx]

    @property
    def P_hat(self):
        return self.P_hat_mat[self.action][self.state.idx]

    @P_hat.setter
    def P_hat(self, new_val):
        self.P_hat_mat[self.action][self.state.idx] = new_val

    def UpdateP(self, next_s):
        curr_num_of_tran = self.P_hat * self.visitations
        curr_num_of_tran[next_s.idx] += 1

        new_est_p_row = curr_num_of_tran / (self.visitations + 1)
        self.P_hat = new_est_p_row

    def UpdateVisits(self):
        self.visitations += 1


class SimulatedState:
    """
    Represents a state with a list of possible actions from current state
    """

    def __init__(self, state, action_num, best_action, P_hat, r_hat_vec):
        self.state = state
        self.P_hat_mat = P_hat
        self.r_hat_vec = r_hat_vec
        self.actions = [StateActionPair(state, action, P_hat) for action in range(action_num)]
        self.policy_action = best_action
        self.r_hat = r_hat_vec[self.state.idx]

    @property
    def idx(self):
        return self.state.idx

    @property
    def r_hat(self):
        return self.r_hat_vec[self.state.idx]

    @r_hat.setter
    def r_hat(self, new_val):
        self.r_hat_vec[self.state.idx] = new_val

    def UpdateReward(self, next_s, new_reward):
        new_val = (self.r_hat * next_s.visitations + new_reward) / (next_s.visitations + 1)
        self.r_hat = new_val

    @property
    def visitations(self):
        return reduce(lambda a, b: a + b, [state_action.visitations for state_action in self.actions])


class Agent:
    def __init__(self, idx, init_state: SimulatedState):
        self.idx = idx
        self.curr_state: SimulatedState = init_state

    def __lt__(self, other):
        return random.choice([True, False])


class Simulator:
    def __init__(self, MDP_model, agent_num=5, init_state=0, gamma=0.9):
        self.MDP_model: MDPModel = MDP_model
        self.r_hat = np.zeros(n)
        self.P_hat = [np.zeros((MDP_model.actions, MDP_model.n)) for _ in range(MDP_model.n)]

        self.policy = [random.randint(0, self.MDP_model.actions - 1) for _ in range(MDP_model.n)]
        self.gamma = gamma
        self.V_hat = np.zeros(n)
        self.states = [SimulatedState(state=state, best_action=self.policy[state.idx],
                                      action_num=self.MDP_model.actions, P_hat=self.P_hat, r_hat_vec=self.r_hat)
                       for state in self.MDP_model.s]
        self.graded_states = {state.idx: random.random() for state in self.states}
        self.agents = Q.PriorityQueue()
        # TODO - Random init_state
        [self.agents.put(PrioritizedObject(Agent(i, self.states[init_state]), i)) for i in range(agent_num)]

    def ImprovePolicy(self):
        self.policy = [random.randint(0, self.MDP_model.actions) for _ in range(self.MDP_model.n)]

    def update_V(self, idx, action):
        self.V_hat[idx] = self.r_hat[idx] + self.gamma * np.dot(self.P_hat[action][idx, :], self.V_hat)

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
        self.ReGradeAllAgents()

    # invoked after states re-prioritization. Replaces queue
    def ReGradeAllAgents(self):
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

    # simulate one action of an agent, and re-grade it, according to it's new state
    def SimulateAgent(self, agent: Agent):
        action = self.policy[agent.curr_state.idx]
        state_action = agent.curr_state.actions[action]

        next_state = np.random.choice(self.states, p=self.MDP_model.P[action][agent.curr_state.idx])
        reward = self.MDP_model.GetReward(state_action.state)
        agent.curr_state.UpdateReward(next_state, reward)
        state_action.UpdateP(next_state)

        self.update_V(state_action.state.idx, state_action.action)
        state_action.UpdateVisits()
        agent.curr_state = agent.curr_state

        self.agents.put(self.GradeAgent(agent))

    def evaluate_P_hat(self):
        return self.P_hat_sum_diff()

    # def EvaluateV(self):
    #     return self.V_hat_diff()


if __name__ == '__main__':
    n = 10
    k = 4

    MDP = MDPModel(n=n)
    #   random_simulator = Simulator(MDP)
    gittins_simulator = Simulator(MDP)

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
