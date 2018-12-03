import numpy as np
import random


class MDPModel:
    def __init__(self, n=10, actions=5):
        self.n: int = n
        self.actions: int = actions
        self.init_prob = self.GenInitialProbability()
        self.P = [np.array([self.gen_P_matrix(state_idx, self.get_succesors(state_idx, action))
                            for action in range(self.actions)]) for state_idx in range(self.n)]

        self.r = self.gen_r_mat()

    def get_succesors(self, state_idx, action):
        return set(range(0, self.n))

    def gen_P_matrix(self, state_idx, succesors):
        if self.IsSinkState(state_idx):
            self_vec = np.zeros(self.n)
            self_vec[state_idx] = 1
            return np.array(self_vec)
        else:
            return np.array(self.gen_row_of_P(succesors))

    def gen_row_of_P(self, succesors):
        row = np.random.random(self.n)  # TODO: variance doesn't calculated as well
        for idx in set(range(self.n)).difference(succesors):
            row[idx] = 0
        row /= row.sum()
        return row

    def gen_r_mat(self):
        return [[(random.random(), random.random()) for _ in range(self.actions)] for _ in range(self.n)]

    def IsSinkState(self, state_idx) -> bool:
        return False

    def GetReward(self, state, action):
        params = self.r[state.idx][action]
        return np.random.normal(params[0], params[1])

    def GenInitialProbability(self):
        return np.ones(self.n) / self.n


class RandomSinkMDP(MDPModel):
    def __init__(self):
        super().__init__()
        self.sink_list = random.sample(range(self.n), random.randint(0, self.n))

    def IsSinkState(self, state_idx):
        return state_idx in self.sink_list


class SeperateChainsMDP(MDPModel):
    def __init__(self, n=10, actions=5, init_states_idx=frozenset({0}), reward_param=((0, 1), (10, 1))):
        if n % 2 == 0:
            n += 1  # make sure sub_chains are even sized

        self.init_states_idx = init_states_idx
        self.chains = [frozenset(range(1, int(n / 2) + 1)), frozenset(range(int(n / 2) + 1, n))]
        self.reward_params = reward_param

        super().__init__(n, actions)

    def FindChain(self, state_idx):
        if state_idx in self.init_states_idx:
            return None
        elif state_idx < self.n / 2:
            return 0
        return 1

    def get_succesors(self, state_idx, action):
        chain = self.FindChain(state_idx)
        if chain is None:
            return self.chains[action]
        return self.chains[chain]

    def GenInitialProbability(self):
        init_prob = np.zeros(self.n)
        for state in self.init_states_idx:
            init_prob[state] = 1 / len(self.init_states_idx)

        return init_prob

    def gen_r_mat(self):
        res = [[] for _ in range(self.n)]
        for state_idx in range(self.n):
            chain = self.FindChain(state_idx)
            for action in range(self.actions):
                if chain is None:
                    params = (0, 0)
                else:
                    params = self.reward_params[chain]
                res[state_idx].append(params)

        return res


class EyeMDP(MDPModel):
    def get_succesors(self, state_idx, action):
        return {np.mod(state_idx + 1, self.n)}


class SingleLineMDP(MDPModel):
    def gen_r_mat(self):
        r_mat = np.zeros((self.n, self.actions))
        r_mat[self.n - 2][0] = 1
        return r_mat

    def IsSinkState(self, state_idx):
        return state_idx == (self.n - 1)

    def get_succesors(self, state_idx, action):
        if action == 0:  # forward -->
            return {(state_idx + 1) % self.n}
        return {0}


if __name__ == '__main__':
    eye_model = SingleLineMDP(n=4, actions=2)
    print(eye_model.P)
