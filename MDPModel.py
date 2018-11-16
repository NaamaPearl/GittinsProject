import numpy as np
import copy
from collections import namedtuple
import queue as Q
from gittins import Gittins
import random


class MDPModel(object):
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

    def simulate_one_step(self, curr_agent):
        curr_s = self.s[curr_agent.curr_state]
        next_s = self.choose_next_state(curr_s)
        r = self.r[curr_s.idx]

        self.update_reward(curr_s, r)
        self.update_p(curr_s, next_s)
        self.update_V(curr_s.idx)
        curr_agent.update_state(next_s.idx)
        curr_s.update_visits()

        return r, next_s.idx

    def choose_next_state(self, curr_s):
        next_s_idx = np.random.choice(np.arange(self.n), p=self.P[curr_s.idx])
        return self.s[next_s_idx]

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

    def get_state(self, n):
        return self.s[n]

    def update_V(self, idx):
        self.V_hat[idx] = self.r_hat[idx] + self.gamma * np.dot(self.P_hat[idx, :], self.V_hat)

    def P_hat_sum_diff(self):
        return abs(self.P - self.P_hat).mean()

    def V_hat_diff(self):
        V = np.dot(self.r, np.linalg.inv(np.eye(self.n) - self.gamma * self.P))
        print(V)
        return abs(self.V_hat - V).max()


class State(object):
    def __init__(self, idx):
        self.idx = idx
        self.visits = 0

    def __repr__(self):
        return str(self.idx)

    def update_visits(self):
        self.visits += 1


class Agent(object):
    def __init__(self, idx, init_state):
        self.idx = idx
        self.curr_state = init_state

    def update_state(self, new_idx_state):
        self.curr_state = new_idx_state

    def __lt__(self, other):
        return random.choice([True, False])


GradedAgent = namedtuple('GradedAgent', ('grade', 'agent'))


class Simulator(object):
    def __init__(self, MDP_model, k=5, init_state=0):
        self.MDP_model = copy.deepcopy(MDP_model)
        self.k = k
        self.agents = Q.PriorityQueue()
        [self.agents.put(GradedAgent(i, Agent(i, init_state))) for i in range(k)]
        self.graded_s = {state.idx: state.idx for state in self.MDP_model.s}

    def GradeStates(self):
        pass

    def ApproxModel(self):
        self.GradeStates()
        self.ReGradeAgents()

    def ReGradeAgents(self):
        new_queue = Q.PriorityQueue()
        while self.agents.qsize() > 0:
            new_queue.put(self.GradeAgent(self.agents.get().agent))

        self.agents = new_queue

    def GradeAgent(self, agent):
        return GradedAgent(self.graded_s[agent.curr_state], agent)

    def simulate(self, steps=10000, grades_freq=20):
        for i in range(steps):
            if i % grades_freq == grades_freq - 1:
                self.ApproxModel()  # prioritize agents & states

            self.SimulateOneStep()

    # find top-priority agent, and activate it for one step
    def SimulateOneStep(self):
        next_agent = self.agents.get()
        self.MDP_model.simulate_one_step(next_agent.agent)  # TODO - move to sim
        self.agents.put(self.GradeAgent(next_agent.agent))

    def evaluate_P_hat(self):
        return self.MDP_model.P_hat_sum_diff()

    def EvaluateV(self):
        return self.MDP_model.V_hat_diff()


class GittinsSimulator(Simulator):
    def GradeStates(self):
        self.graded_s = Gittins(self.MDP_model)

    def evaluateGittins(self):
        real_gittins = Gittins(self.MDP_model, approximation=False)
        print('evaluate: ' + str(self.graded_s))
        print('real: ' + str(real_gittins))
        return sum(np.not_equal(real_gittins, self.graded_s))


class RandomSimulator(Simulator):
    def GradeStates(self):
        return {state: random.random() for state in self.MDP_model.s}


if __name__ == '__main__':
    n = 4
    k = 2

    MDP = MDPModel(n=n)
    RandomSimulator = RandomSimulator(k=k, MDP_model=MDP)
    GittinsSimulator = GittinsSimulator(MDP_model=MDP, k=k)

    # RandomSimulator.simulate(steps=100000)
    GittinsSimulator.simulate(steps=10000)

    print('eval Random')
    print(RandomSimulator.evaluate_P_hat())
    print(RandomSimulator.EvaluateV())

    print('eval Gittin')
    print(GittinsSimulator.evaluate_P_hat())
    print(GittinsSimulator.EvaluateV())

    # print(GittinsSimulator.evaluateGittins())
    print('all done')
