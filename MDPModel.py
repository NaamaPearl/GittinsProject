import numpy as np
import random


class MDPModel:
    def __init__(self, n=10, actions=5):
        self.n: int = n
        self.actions: int = actions
        self.init_prob = self.GenInitialProbability()
        for state_idx in range(self.n):
            low, high = self.ChainLimits(state_idx)
            self.P = [self.gen_P_matrix(state_idx, set(range(low, high + 1))) for _ in range(self.n)]
        self.r = self.gen_r_vec()

    def ChainLimits(self, state_idx):
        return 0, self.n

    def gen_P_matrix(self, state_idx, succesors):
        if self.IsSinkState(state_idx):
            self_vec = np.zeros(self.n)
            self_vec[state_idx] = 1
            return np.array([self_vec for _ in range(self.actions)])
        else:
            return np.array([self.gen_row_of_P(self.n, succesors) for _ in range(self.actions)])

    @staticmethod
    def gen_row_of_P(n, succesors):
        row = np.random.random(n)  # TODO: variance doesn't calculated as well
        for idx in set(range(n)).difference(succesors):
            row[idx] = 0
        row /= row.sum()
        return row

    def gen_r_vec(self):
        return np.random.random_integers(low=0, high=100, size=(self.n, self.actions))  # TODO stochastic reward

    def IsSinkState(self, state_idx) -> bool:
        return False

    def GetReward(self, state, action):
        return self.r[state.idx, action]

    def GenInitialProbability(self):
        return np.ones(self.n) / self.n


class RandomSinkMDP(MDPModel):
    def __init__(self):
        super().__init__()
        self.sink_list = random.sample(range(self.n), random.randint(0, self.n))

    def IsSinkState(self, state_idx):
        return state_idx in self.sink_list


class SeperateChainsMDP(MDPModel):
    def __init__(self, init_states_idx=frozenset({0})):
        super().__init__()
        self.init_states_idx = init_states_idx

    def ChainLimits(self, state_idx):
        if state_idx in self.init_states_idx:
            return 0, self.n
        elif state_idx < self.n / 2:
            return 0, self.n / 2 - 1
        return self.n / 2, self.n

    def GenInitialProbability(self):
        self.init_prob = np.zeros(self.n)
        for state in self.init_states_idx:
            self.init_prob[state] = 1 / len(self.init_states_idx)
