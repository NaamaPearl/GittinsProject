import numpy as np
import random


class MDPModel:
    def __init__(self, n, actions, chain_num=1):
        self.n: int = n
        self.chain_num = chain_num
        self.actions: int = actions
        self.init_prob = self.GenInitialProbability()
        self.P = [np.array([self.gen_P_matrix(state_idx, self.get_succesors(state_idx, action))
                            for action in range(self.actions)]) for state_idx in range(self.n)]

        self.r = self.gen_r_mat()

    def FindChain(self, state_idx):
        return 0

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

    def GetReward(self, state, action, policy, time_to_run=1):
        position_vec = np.zeros(self.n)
        position_vec[state.idx] = 1
        params = self.r[state.idx][action]
        reward = np.random.normal(params[0], params[1])

        if time_to_run > 1:
            policy_dynamic = self.GetPolicyDynamics(policy)
            expected_policy_rewards = self.GetPolicyExpectedRewards(policy)
            for _ in range(time_to_run - 1):  # first simulation is made in the previous row
                position_vec = policy_dynamic @ position_vec
                reward += (position_vec @ expected_policy_rewards)

        return reward

    def GetPolicyExpectedRewards(self, policy):
        return np.array([self.r[i][a][0] for (i, a) in enumerate(policy)])

    def GetPolicyDynamics(self, policy):
        return np.array([self.P[i][a] for (i, a) in enumerate(policy)])

    def GenInitialProbability(self):
        return np.ones(self.n) / self.n


class RandomSinkMDP(MDPModel):
    def __init__(self, n, actions):
        self.sink_list = random.sample(range(n), random.randint(0, n))
        super().__init__(n, actions)

    def IsSinkState(self, state_idx):
        return state_idx in self.sink_list


class SeperateChainsMDP(MDPModel):
    def __init__(self, n, reward_param, init_states_idx=frozenset({0})):
        self.chain_num = len(reward_param)
        self.init_states_idx = init_states_idx
        n += (1 - n % self.chain_num)  # make sure sub_chains are even sized
        self.chain_size = int((n - 1) / self.chain_num)

        self.chains = [set(range(1 + i * self.chain_size, (i + 1) * self.chain_size + 1))
                       for i in range(self.chain_num)]
        self.reward_params = reward_param

        super().__init__(n, self.chain_num, self.chain_num)

    def FindChain(self, state_idx):
        if state_idx in self.init_states_idx:
            return None
        for i in range(self.chain_num):
            if state_idx < 1 + (i + 1) * self.chain_size:
                return i

    def get_succesors(self, state_idx, action):
        chain = self.FindChain(state_idx)
        if chain is None:
            return self.chains[action]
        return self.chains[chain]

    def GenInitialProbability(self):
        init_prob = np.ones(self.n)
        for state in self.init_states_idx:
            init_prob[state] = 0

        return init_prob / sum(init_prob)

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
