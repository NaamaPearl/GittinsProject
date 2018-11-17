import numpy as np
import copy
from collections import namedtuple
import queue as Q
from gittins import Gittins
import random


class MDPModel:
    def __init__(self, n=10, gamma=0.9, loc=0.5, var=0.2):
        self.n = n
        self.P = self.gen_P_matrix(loc, var)
        self.P_hat = np.zeros((n, n))
        self.r = np.random.random_integers(low=0, high=100, size=n)  # TODO stochastic reward
        self.r_hat = np.zeros(n)
        self.s = [State(i) for i in range(n)]
        self.V_hat = np.zeros(n)
        self.gamma = gamma

    def gen_P_matrix(self, loc, var):
        P = np.array([self.get_row_of_P(self.n, loc, var) for _ in range(self.n)])
        return P

    @staticmethod
    def get_row_of_P(n, loc, var):
        row = np.random.random(n)  # TODO: variance doesn't calculated as well
        row /= row.sum()
        return row

    def update_reward(self, next_s, new_reward):
        curr_est_reward = self.r_hat[next_s.idx]
        new_est_reward = (curr_est_reward * next_s.visits + new_reward) / (next_s.visits + 1)

        self.r_hat[next_s.idx] = new_est_reward

    def update_p(self, curr_s, next_s):
        curr_est_p_row = self.P_hat[curr_s.idx]
        curr_num_of_tran = curr_est_p_row * curr_s.visits
        curr_num_of_tran[next_s.idx] += 1

        new_est_p_row = curr_num_of_tran / (curr_s.visits + 1)
        self.P_hat[curr_s.idx] = new_est_p_row

    def update_V(self, idx):
        self.V_hat[idx] = self.r_hat[idx] + self.gamma * np.dot(self.P_hat[idx, :], self.V_hat)

    def P_hat_sum_diff(self):
        return abs(self.P - self.P_hat).mean()

    def V_hat_diff(self):
        V = np.dot(self.r, np.linalg.inv(np.eye(self.n) - self.gamma * self.P))
        print(V)
        return abs(self.V_hat - V).max()


class State:
    def __init__(self, idx):
        self.idx = idx
        self.visits = 0

    # def __repr__(self):
    #     return 'state' + str(self.idx)

    def update_visits(self):
        self.visits += 1


class Agent:
    def __init__(self, idx, init_state):
        self.idx = idx
        self.curr_state = init_state

    def __lt__(self, other):
        return random.choice([True, False])


GradedAgent = namedtuple('GradedAgent', ('grade', 'agent'))


class Simulator:
    def __init__(self, MDP_model, agent_num=5, init_state=0, ):
        self.MDP_model = copy.deepcopy(MDP_model)
        self.graded_states = {state.idx: random.random() for state in self.MDP_model.s}
        self.agents = Q.PriorityQueue()
        [self.agents.put(GradedAgent(i, Agent(i, init_state))) for i in range(agent_num)]  # TODO - Random init_state

    def GradeStates(self):
        pass

    def ApproxModel(self):
        self.GradeStates()
        self.ReGradeAllAgents()

    # invoked after states re-prioritization. Replaces queue
    def ReGradeAllAgents(self):
        new_queue = Q.PriorityQueue()
        while self.agents.qsize() > 0:
            new_queue.put(self.GradeAgent(self.agents.get().agent))

        self.agents = new_queue

    def GradeAgent(self, agent):
        return GradedAgent(self.graded_states[agent.curr_state], agent)

    def simulate(self, steps=10000, grades_freq=20):
        for i in range(steps):
            self.SimulateOneStep()
            if i % grades_freq == grades_freq - 1:
                self.ApproxModel()  # prioritize agents & states

    # find top-priority agents, and activate them for a single step
    def SimulateOneStep(self, agents_to_run=1):
        agents_list = [self.agents.get().agent for _ in range(agents_to_run)]
        for agent in agents_list:
            self.SimulateAgent(agent)

    # simulate one action of an agent, and re-grade it, according to it's new state
    def SimulateAgent(self, agent):
        curr_s = self.MDP_model.s[agent.curr_state]
        next_s = self.choose_next_state(curr_s)
        r = self.MDP_model.r[curr_s.idx]

        self.MDP_model.update_reward(curr_s, r)
        self.MDP_model.update_p(curr_s, next_s)
        self.MDP_model.update_V(curr_s.idx)
        agent.curr_state = next_s.idx
        curr_s.update_visits()

        self.agents.put(self.GradeAgent(agent))

    def choose_next_state(self, curr_s):
        next_s_idx = np.random.choice(np.arange(self.MDP_model.n), p=self.MDP_model.P[curr_s.idx])
        return self.MDP_model.s[next_s_idx]

    def evaluate_P_hat(self):
        return self.MDP_model.P_hat_sum_diff()

    def EvaluateV(self):
        return self.MDP_model.V_hat_diff()


class GittinsSimulator(Simulator):
    def GradeStates(self):
        self.graded_states = Gittins(self.MDP_model)

    def evaluateGittins(self):
        real_gittins = Gittins(self.MDP_model, approximation=False)
        print('evaluate: ' + str(self.graded_states))
        print('real: ' + str(real_gittins))
        return sum(np.not_equal(real_gittins, self.graded_states))


class RandomSimulator(Simulator):
    def GradeStates(self):
        self.graded_states = {state.idx: random.random() for state in self.MDP_model.s}


if __name__ == '__main__':
    n = 10
    k = 4

    MDP = MDPModel(n=n)
    RandomSimulator = RandomSimulator(agent_num=k, MDP_model=MDP)
    GittinsSimulator = GittinsSimulator(MDP_model=MDP, agent_num=k)

    RandomSimulator.simulate(steps=100000)
    GittinsSimulator.simulate(steps=10000)

    print('eval Random')
    print(RandomSimulator.evaluate_P_hat())
    print(RandomSimulator.EvaluateV())

    print('eval Gittin')
    print(GittinsSimulator.evaluate_P_hat())
    print(GittinsSimulator.EvaluateV())

    # print(GittinsSimulator.evaluateGittins())
    print('all done')
