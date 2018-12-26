import numpy as np
import random
from functools import reduce
from MDPModel.RewardGenerator import RewardGeneratorFactory

threshold = 10 ** -3


class MDPModel:
    def __init__(self, n, actions, chain_num, gamma, traps_num=0):
        self.n: int = n
        self.type = 'regular'
        self.chain_num = chain_num
        self.actions: int = actions
        self.init_prob = self.GenInitialProbability()

        self.traps = list(np.random.choice(list(self.GetActiveChains()), traps_num))
        self.leads_to_trap = {}
        self.P = self.BuildP()

        self.r = self.gen_r_mat()
        self.expected_r = np.array([[self.r[s][a].expected_reward for s in range(self.n)] for a in range(self.actions)])
        self.gamma = gamma
        self.opt_policy = self.CalcOptPolicy()

    def BuildP(self, **kwargs):
        possible_successors = self.GenPossibleSuccessors()

        return [np.array([self.gen_P_matrix(state_idx,
                                            self.get_successors(state_idx, action, possible_successors, **kwargs))
                          for action in range(self.actions)]) for state_idx in range(self.n)]

    def GenPossibleSuccessors(self, **kwargs):
        return [list(range(self.n))]

    def GetActiveChains(self):
        return list(range(self.n))

    def IsStateActionRewarded(self, state, action):
        return True

    def FindChain(self, state_idx):
        return 0

    def get_successors(self, state_idx, action, possible_successors, **kwargs):
        return set(random.sample(possible_successors[state_idx], self.succ_num))

    def gen_P_matrix(self, state_idx, succesors):
        if self.IsSinkState(state_idx):
            self_vec = np.zeros(self.n)
            self_vec[state_idx] = 1
            return np.array(self_vec)
        else:
            return np.array(self.gen_row_of_P(succesors, state_idx))

    def gen_row_of_P(self, succesors, state_idx):
        row = np.array([random.random() if idx in succesors else 0 for idx in range(self.n)])
        return row / sum(row)

    def GetRewardParams(self, state_idx, action):
        return None

    def gen_r_mat(self):
        res = [[] for _ in range(self.n)]
        for state_idx in range(self.n):
            res[state_idx] = [RewardGeneratorFactory.Generate(self.IsStateActionRewarded(state_idx, act),
                                                              reward_params=self.GetRewardParams(state_idx, act))
                              for act in range(self.actions)]

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
                V_new = r + (self.P[s] @ (self.gamma * V_old))
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
        if 'offline' in params['eval_type']:
            return reduce((lambda x, y: x + y),
                          [self.opt_r @ (self.init_prob @ np.linalg.matrix_power(self.opt_P, i))
                           for i in range(params['trajectory_len'])])
        if 'online' in params['eval_type']:
            expected_reward_vec = [self.opt_r @ (self.init_prob @ np.linalg.matrix_power(self.opt_P, i))
                                   for i in range(params['steps'])]
            batch_reward = [sum(group)
                            for group in np.array_split(expected_reward_vec, params['steps'] / params['eval_freq'])]
            return np.cumsum(batch_reward)

        raise ValueError('unexpected evaluation type')

class RandomSinkMDP(MDPModel):
    def __init__(self, n, actions, chain_num, gamma):
        self.sink_list = random.sample(range(n), random.randint(0, n))
        super().__init__(n, actions=actions, chain_num=chain_num, gamma=gamma)
        self.type = 'random_sink'

    def IsSinkState(self, state_idx):
        return state_idx in self.sink_list


class SeperateChainsMDP(MDPModel):
    def __init__(self, n, action, succ_num, reward_param, gamma, chain_num, op_succ_num,
                 traps_num=0, init_states_idx=frozenset({0})):
        self.chain_num = chain_num
        self.init_states_idx = init_states_idx
        n += (1 - n % self.chain_num)  # make sure sub_chains are even sized
        self.chain_size = int((n - 1) / self.chain_num)

        self.chains = [set(range(1 + i * self.chain_size, (i + 1) * self.chain_size + 1))
                       for i in range(self.chain_num)]
        self.reward_params = reward_param
        self.succ_num = succ_num
        self.op_succ_num = op_succ_num

        super().__init__(n, actions=action, chain_num=self.chain_num, gamma=gamma, traps_num=traps_num)
        self.type = 'chains'

    # def IsSinkState(self, state_idx):
    #     return state_idx in self.traps

    # def BuildP(self):
    #     succ_list = []
    #     for state_idx in range(self.n):
    #         chain = self.FindChain(state_idx)
    #         if chain is None:
    #             succ_list.append([])
    #         else:
    #             succ_list.append(set(np.random.choice(list(self.chains[self.FindChain(state_idx)]), self.op_succ_num)))
    #     return super().BuildP(succ_list=succ_list)

    def GenPossibleSuccessors(self, **kwargs):
        try:
            forbidden_states = set(kwargs['forbidden_states']).union(self.init_states_idx)
        except KeyError:
            forbidden_states = self.init_states_idx

        possible_per_chain = [chain.difference(forbidden_states) for chain in self.chains]
        return [self.GenPossibleSuccessorsPerState(self.FindChain(s), possible_per_chain) for s in range(self.n)]

    def GenPossibleSuccessorsPerState(self, chain_num, possible_per_chain):
        if chain_num is None:
            successors = [set(opt) for opt in possible_per_chain]
        else:
            successors = set(random.sample(possible_per_chain[chain_num], self.op_succ_num))

        return successors

    def get_successors(self, state_idx, action, possible_successors, **kwargs):
        if state_idx in self.init_states_idx:
            max_states = min([len(chain_states) for chain_states in possible_successors[state_idx]])
            return reduce(lambda a, b: a.union(b), [set(random.sample(chain_succ, max_states))
                                                    for chain_succ in possible_successors[state_idx]])
        return super().get_successors(state_idx, action, possible_successors, **kwargs)

    def IsStateActionRewarded(self, state_idx, action):
        return self.FindChain(state_idx) not in [None, 0]

    def FindChain(self, state_idx):
        if state_idx in self.init_states_idx:
            return None
        for i in range(self.chain_num):
            if state_idx < 1 + (i + 1) * self.chain_size:
                return i

    def GenInitialProbability(self):
        init_prob = np.zeros(self.n)
        for state in self.init_states_idx:
            init_prob[state] = 1

        return init_prob / sum(init_prob)

    def GetRewardParams(self, state_idx, action):
        chain = self.FindChain(state_idx)
        if chain is None:
            return None

        if state_idx in self.traps:
            return self.reward_params['trap']

        if (state_idx, action) in self.leads_to_trap:
            return self.reward_params['leads_to_trap']

        return self.reward_params.get(chain)

    def gen_row_of_P(self, succesors, state_idx):
        if state_idx in self.init_states_idx:
            res = np.array([1 if state in succesors else 0 for state in range(self.n)])
            return res / sum(res)

        return super().gen_row_of_P(succesors, state_idx)

    def CalcOptExpectedReward(self, general_sim_params):
        return self.chain_num * super().CalcOptExpectedReward(general_sim_params)

    def GetActiveChains(self):
        return self.chains[1:]


class ChainsTunnelMDP(SeperateChainsMDP):
    def __init__(self, n, action, succ_num, reward_param, gamma, chain_num, op_succ_num, tunnel_indexes, traps_num=0):
        self.tunnel_indexes = tunnel_indexes
        super().__init__(n, action, succ_num, reward_param, gamma, chain_num, op_succ_num, traps_num)

    def IsStateActionRewarded(self, state_idx, action):
        if state_idx == self.tunnel_indexes[-1]:
            return action == 0

        return super().IsStateActionRewarded(state_idx, action)

    def get_successors(self, state_idx, action, possible_successors, **kwargs):
        if state_idx in self.tunnel_indexes[:-1]:
            if action == 0:
                return {self.tunnel_indexes[state_idx - self.tunnel_indexes[0] + 1]}
            return {self.tunnel_indexes[0]}

        return super().get_successors(state_idx, action, possible_successors, **kwargs)

    def GenPossibleSuccessors(self, **kwargs):
        kwargs['forbidden_states'] = self.tunnel_indexes[1:]
        return super().GenPossibleSuccessors(**kwargs)

    def GetRewardParams(self, state_idx, act):
        if state_idx == self.tunnel_indexes[-1]:
            return self.reward_params['tunnel_end']
        if state_idx in self.tunnel_indexes:
            return self.reward_params['lead_to_tunnel']
        return super().GetRewardParams(state_idx, act, )


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
    def get_successors(self, state_idx, action, **kwargs):
        return {np.mod(state_idx + 1, self.n)}


class SingleLineMDP(MDPModel):
    def IsStateActionRewarded(self, state_idx, action):
        return state_idx == self.n - 2 and action == 0

    def get_successors(self, state_idx, action, **kwargs):
        if action == 0:  # forward -->
            return {(state_idx + 1) % self.n}
        return {0}


if __name__ == "__main__":
    sigline = SingleLineMDP(n=5,
                            actions=3,
                            chain_num=1,
                            gamma=0.9,
                            traps_num=0)

    seperate = SeperateChainsMDP(
        n=31,
        action=4,
        succ_num=2,
        chain_num=2,
        gamma=0.9,
        reward_param={1: {'bernoulli_p': 1, 'gauss_params': ((10, 3), 1)}},
    )

    print('doen')
