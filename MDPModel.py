import numpy as np
import random


class MDPModel:
    def __init__(self, n=10, actions=5):
        self.n: int = n
        self.actions: int = actions
        self.P = [self.gen_P_matrix() for _ in range(n)]
        self.r = np.random.random_integers(low=0, high=100, size=n)  # TODO stochastic reward
        self.s = [State(i) for i in range(n)]

    def gen_P_matrix(self):
        P = np.array([self.get_row_of_P(self.n) for _ in range(self.actions)])
        return P

    @staticmethod
    def get_row_of_P(n):
        row = np.random.random(n)  # TODO: variance doesn't calculated as well
        row /= row.sum()
        return row

    def GetReward(self, state):
        return self.r[state.idx]


class State:
    def __init__(self, idx):
        self.idx = idx
        self.visits = 0

    def update_visits(self):
        self.visits += 1

    def __hash__(self):
        return hash(self.idx)

