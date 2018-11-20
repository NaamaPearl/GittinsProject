import numpy as np
import random


class MDPModel:
    def __init__(self, n=10, actions=5, gamma=0.9, loc=0.5, var=0.2):
        self.n: int = n
        self.actions: int = actions
        self.P = [self.gen_P_matrix(loc, var) for _ in range(actions)]
        self.P_hat = [np.zeros((n, n)) for _ in range(actions)]
        self.r = [np.random.random_integers(low=0, high=100, size=n)for _ in range(actions)]  # TODO stochastic reward
        self.r_hat = [np.zeros(n) for _ in range(actions)]
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

    def update_reward(self, action, next_s, new_reward):
        curr_est_reward = self.r_hat[action][next_s.idx]
        new_est_reward = (curr_est_reward * next_s.visits + new_reward) / (next_s.visits + 1)

        self.r_hat[action][next_s.idx] = new_est_reward

    def update_p(self, curr_s, action, next_s):
        curr_est_p_row = self.P_hat[action][curr_s.idx]
        curr_num_of_tran = curr_est_p_row * curr_s.visits
        curr_num_of_tran[next_s.idx] += 1

        new_est_p_row = curr_num_of_tran / (curr_s.visits + 1)
        self.P_hat[action][curr_s.idx] = new_est_p_row

    def update_V(self, idx, action):
        self.V_hat[idx] = self.r_hat[action][idx] + self.gamma * np.dot(self.P_hat[action][idx, :], self.V_hat)

    def P_hat_sum_diff(self):
        return [abs(self.P[a] - self.P_hat[a]).mean() for a in range(self.actions)]

    def V_hat_diff(self):
        res =[]
        for action in range(self.actions):
            V = np.dot(self.r, np.linalg.inv(np.eye(self.n) - self.gamma * self.P[action]))
            print(V)
            res.append(abs(self.V_hat - V).max())


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


class PrioritizedObject:
    """
    Represents an object with a prioritization
    """
    def __init__(self, obj, r=0):
        self.object = obj
        self.gittins = r

    def __gt__(self, other):
        return self.gittins > other.gittins