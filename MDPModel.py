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
        self.init_states_idx = []

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
            return np.array(self.gen_row_of_P(succesors, state_idx))

    def gen_row_of_P(self, succesors, state_idx):
        row = np.random.random(self.n)
        for idx in set(range(self.n)).difference(succesors):
            row[idx] = 0
        row /= row.sum()
        return row

    def gen_r_mat(self):
        return [[(random.random(), random.random()) for _ in range(self.actions)] for _ in range(self.n)]

    def IsSinkState(self, state_idx) -> bool:
        return False

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
        if state_idx == 0:
            return {np.random.choice(list(chain)) for chain in self.chains}
        chain = self.FindChain(state_idx)
        if chain is None:
            return self.chains[action]
        return self.chains[chain]

    def GenInitialProbability(self):
        init_prob = np.zeros(self.n)
        for state in self.init_states_idx:
            init_prob[state] = 1

        return init_prob / sum(init_prob)

    def gen_r_mat(self):
        res = [[] for _ in range(self.n)]
        for state_idx in range(self.n):
            chain = self.FindChain(state_idx)
            for action in range(self.actions):
                if chain is None:
                    params = (0, 0)
                else:
                    expectation = np.random.normal(self.reward_params[chain][0], self.reward_params[chain][1])
                    variance = self.reward_params[chain][2]
                    params = (expectation, variance)
                res[state_idx].append(params)

        return res


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
        r_mat = np.zeros((self.n, self.actions))
        r_mat[self.n - 2][0] = 1
        return r_mat

    def IsSinkState(self, state_idx):
        return state_idx == (self.n - 1)

    def get_succesors(self, state_idx, action):
        if action == 0:  # forward -->
            return {(state_idx + 1) % self.n}
        return {0}


class SimulatedModel:
    def __init__(self, mdp_model):
        self.MDP_model: MDPModel = mdp_model
        self.policy_dynamics = np.zeros((mdp_model.n, mdp_model.n))
        self.policy_expected_rewards = np.zeros(mdp_model.n)

    def CalcPolicyData(self, policy):
        for i, a in enumerate(policy):
            self.policy_dynamics[i] = self.MDP_model.P[i][a]
            self.policy_expected_rewards[i] = self.MDP_model.r[i][a][0]

    def GetNextState(self, state_action, run_time=1):
        n_s = np.random.choice(range(self.MDP_model.n), p=self.MDP_model.P[state_action.state.idx][state_action.action])
        if run_time == 1:
            return n_s

        p = self.policy_dynamics ** (run_time - 1)
        return np.random.choice(range(self.MDP_model.n), p=p[n_s])

    def GetReward(self, state_action, gamma, time_to_run=1):
        params = self.MDP_model.r[state_action.state.idx][state_action.action]
        reward = np.random.normal(params[0], params[1])

        if time_to_run > 1:
            position_vec = np.zeros(self.MDP_model.n)
            position_vec[state_action.state.idx] = 1

            for i in range(time_to_run - 1):  # first simulation is made in the previous row
                position_vec = self.policy_dynamics @ position_vec
                reward += (gamma ** i * (position_vec @ self.policy_expected_rewards))

        return reward

    def calculate_V(self, gamma):
        return np.linalg.inv(np.eye(self.MDP_model.n) - gamma * self.policy_dynamics) @ self.policy_expected_rewards

    @property
    def n(self):
        return self.MDP_model.n

    @property
    def actions(self):
        return self.MDP_model.actions

    @property
    def init_prob(self):
        return self.MDP_model.init_prob

    def FindChain(self, idx):
        return self.MDP_model.FindChain(idx)

    @property
    def chain_num(self):
        return self.MDP_model.chain_num

    @property
    def init_states_idx(self):
        return self.MDP_model.init_states_idx


if __name__ == '__main__':
    eye_model = SingleLineMDP(n=4, actions=2)
    print(eye_model.P)
