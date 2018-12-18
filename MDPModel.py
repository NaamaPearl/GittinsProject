import numpy as np
import random
from functools import reduce
from abc import abstractmethod
from itertools import groupby

threshold = 10 ** -3


class RewardGenerator:
    def __init__(self):
        self.expected_reward = 0

    @abstractmethod
    def GiveReward(self):
        return 0


class RandomRewardGenerator(RewardGenerator):
    def __init__(self, gauss_params, bernoulli_p):
        super().__init__()
        self.bernoulli_p = random.random() if bernoulli_p is None else bernoulli_p
        self.gauss_mu = np.random.normal(gauss_params[0], gauss_params[1])
        self.gauss_sigma = gauss_params[2]
        self.expected_reward = self.gauss_mu * self.bernoulli_p

    def GiveReward(self):
        if np.random.binomial(1, self.bernoulli_p):
            return np.random.normal(self.gauss_mu, self.gauss_sigma)
        return 0


class RewardGeneratorFactory:
    @staticmethod
    def Generate(rewarded_state, reward_params):
        if rewarded_state:
            if reward_params is None:
                return RandomRewardGenerator((random.random(), random.random(), random.random()))
            return RandomRewardGenerator(gauss_params=reward_params['gauss_params'],
                                         bernoulli_p=reward_params.get('bernoulli_p'))
        return RewardGenerator()


class MDPModel:
    def __init__(self, n, actions, chain_num, gamma):
        self.n: int = n
        self.type = 'regular'
        self.chain_num = chain_num
        self.actions: int = actions
        self.init_prob = self.GenInitialProbability()
        self.P = [np.array([self.gen_P_matrix(state_idx, self.get_succesors(state_idx, action))
                            for action in range(self.actions)]) for state_idx in range(self.n)]

        self.r = self.gen_r_mat()
        self.gamma = gamma
        self.opt_policy = self.CalcOptPolicy()

    def IsStateRewarded(self, state_idx):
        return True

    def FindChain(self, state_idx):
        return 0

    def get_succesors(self, state_idx, action):
        return set(range(self.n))

    def gen_P_matrix(self, state_idx, succesors):
        if self.IsSinkState(state_idx):
            self_vec = np.zeros(self.n)
            self_vec[state_idx] = 1
            return np.array(self_vec)
        else:
            return np.array(self.gen_row_of_P(succesors, state_idx))

    def gen_row_of_P(self, succesors, state_idx):
        row = np.zeros(self.n)
        for idx in succesors:
            row[idx] = random.random()
        row /= row.sum()
        return row

    def GetRewardParams(self, state_idx):
        return None

    def gen_r_mat(self, **kwargs):
        res = [[] for _ in range(self.n)]
        for state_idx in range(len(res)):
            is_state_rewarded = self.IsStateRewarded(state_idx)
            reward_params = self.GetRewardParams(state_idx)
            res[state_idx] = [RewardGeneratorFactory.Generate(is_state_rewarded, reward_params)
                              for _ in range(self.actions)]

        return res

    def IsSinkState(self, state_idx) -> bool:
        return False

    def GenInitialProbability(self):
        return np.ones(self.n) / self.n

    def CalcOptPolicy(self):
        V = np.zeros(self.n)
        V_old = np.ones(self.n)
        policy = np.zeros(self.n, dtype=int)
        while any(abs(V - V_old) > threshold):
            V_old = np.copy(V)
            for s in range(self.n):
                r = [rv.expected_reward for rv in self.r[s]]
                V_new = r + self.gamma * (self.P[s] @ V_old)
                V[s] = max(V_new)
                policy[s] = np.argmax(V_new)
        return policy

    @property
    def opt_r(self):
        return np.array([self.r[s][self.opt_policy[s]].expected_reward for s in range(self.n)])

    @property
    def opt_P(self):
        prob_mat = np.zeros((self.n, self.n))
        for state in range(self.n):
            action = self.opt_policy[state]
            prob_mat[state] = self.P[state][action]
        return prob_mat

    def CalcOptExpectedReward(self, params):
        if params['eval_type'] == 'offline':
            return reduce((lambda x, y: x + y),
                          [self.opt_r @ (self.init_prob @ np.linalg.matrix_power(self.opt_P, i))
                           for i in range(params['trajectory_len'])])
        if params['eval_type'] == 'online':
            expected_reward_vec = [self.opt_r @ (self.init_prob @ np.linalg.matrix_power(self.opt_P, i))
                                   for i in range(params['steps'])]
            batch_reward = [sum(group)
                              for group in np.array_split(expected_reward_vec, params['steps'] / params['eval_freq'])]
            return np.cumsum(batch_reward)


class RandomSinkMDP(MDPModel):
    def __init__(self, n, actions, reward_type, chain_num, gamma):
        self.sink_list = random.sample(range(n), random.randint(0, n))
        super().__init__(n, actions, reward_type, chain_num, gamma)
        self.type = 'random_sink'

    def IsSinkState(self, state_idx):
        return state_idx in self.sink_list


class SeperateChainsMDP(MDPModel):
    def __init__(self, n, reward_param, reward_type, gamma, chain_num, init_states_idx=frozenset({0})):
        self.chain_num = chain_num
        self.init_states_idx = init_states_idx
        n += (1 - n % self.chain_num)  # make sure sub_chains are even sized
        self.chain_size = int((n - 1) / self.chain_num)

        self.chains = [set(range(1 + i * self.chain_size, (i + 1) * self.chain_size + 1))
                       for i in range(self.chain_num)]
        self.reward_params = reward_param

        super().__init__(n, actions=self.chain_size, chain_num=self.chain_num, gamma=gamma)
        self.type = 'chains'

    def IsStateRewarded(self, state_idx):
        return self.FindChain(state_idx) not in [None, 0]

    def FindChain(self, state_idx):
        if state_idx in self.init_states_idx:
            return None
        for i in range(self.chain_num):
            if state_idx < 1 + (i + 1) * self.chain_size:
                return i

    def get_succesors(self, state_idx, action):
        chain = self.FindChain(state_idx)
        if chain is None:
            return set(range(self.n))  # self.chains[action]

        return {1 + self.chain_size * chain + action}

    def GenInitialProbability(self):
        init_prob = np.zeros(self.n)
        for state in self.init_states_idx:
            init_prob[state] = 1

        return init_prob / sum(init_prob)

    def GetRewardParams(self, state_idx):
        return self.reward_params.get(self.FindChain(state_idx))

    def gen_row_of_P(self, succesors, state_idx):
        if state_idx in self.init_states_idx:
            res = np.ones(self.n)
            res[0] = 0
            return res / sum(res)

        return super().gen_row_of_P(succesors, state_idx)

    def CalcOptExpectedReward(self, trajectory_len):
        return self.chain_num * super().CalcOptExpectedReward(trajectory_len)

# class StarMDP(SeperateChainsMDP):
#     def get_succesors(self, state_idx, action):
#         if state_idx == 0:
#             return {list(self.chains[(action % self.chain_num)])[0]}
#
#         next_state = state_idx + 1
#         next_chain = self.FindChain(state_idx)
#         curr_chain = self.FindChain(state_idx)
#         if next_chain == curr_chain:
#             return next_state
#         else:
#             return self.chains[curr_chain][0]


class EyeMDP(MDPModel):
    def get_succesors(self, state_idx, action):
        return {np.mod(state_idx + 1, self.n)}


class SingleLineMDP(MDPModel):
    def gen_r_mat(self):
        r_mat = [[(0, 0) for _ in range(self.actions)] for _ in range(self.n)]
        r_mat[self.n - 2][0] = (1, 0)
        return r_mat

    def IsSinkState(self, state_idx):
        return state_idx == (self.n - 1)

    def get_succesors(self, state_idx, action):
        if action == 0:  # forward -->
            return {(state_idx + 1) % self.n}
        return {0}
