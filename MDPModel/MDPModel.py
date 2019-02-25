import numpy as np
import random
from functools import reduce
from MDPModel.RewardGenerator import RewardGeneratorFactory


threshold = 10 ** -10


class MDPModel:
    def __init__(self, n, actions, chain_num, gamma, succ_num, **kwargs):
        self.n: int = n
        self.type = 'regular'
        self.chain_num = chain_num
        self.actions: int = actions
        self.init_prob = self.GenInitialProbability()

        self.succ_num = succ_num
        self.possible_suc = None
        self.P = self.BuildP()

        self.r = self.gen_r_mat()
        self.expected_r = np.array([[self.r[s][a].expected_reward for s in range(self.n)] for a in range(self.actions)])
        self.gamma = gamma
        self.opt_policy, self.V = self.CalcOptPolicy()

    @property
    def active_chains_ratio(self):
        return 1

    def BuildP(self):
        self.possible_suc = self.GenPossibleSuccessors()

        return [np.array([self.gen_P_matrix(state_idx, self.get_successors(state_idx, action=act))
                          for act in range(self.actions)]) for state_idx in range(self.n)]

    def GenPossibleSuccessors(self, **kwargs):
        return [list(range(self.n)) for _ in range(self.n)]

    def GetActiveChains(self):
        return {0}

    def IsStateActionRewarded(self, state, action):
        return True

    def FindChain(self, state_idx):
        return None

    def get_successors(self, state_idx, **kwargs):
        return set(random.sample(self.possible_suc[state_idx], self.succ_num))

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

    def GetRewardParams(self, state_idx):
        return None

    def gen_r_mat(self):
        res = [[] for _ in range(self.n)]
        for state_idx in range(self.n):
            reward_params = self.GetRewardParams(state_idx)
            res[state_idx] = [RewardGeneratorFactory.Generate(self.IsStateActionRewarded(state_idx, act),
                                                              reward_params=reward_params)
                              for act in range(self.actions)]

        return res

    def IsSinkState(self, state_idx):
        return False

    def GenInitialProbability(self):
        return np.ones(self.n) / self.n

    def CalcPolicyData(self, policy):
        policy_dynamics = np.zeros((self.n, self.n))
        policy_expected_rewards = np.zeros(self.n)
        for i, a in enumerate(policy):
            policy_dynamics[i] = self.P[i][a]
            policy_expected_rewards[i] = self.r[i][a].expected_reward

        return policy_dynamics, policy_expected_rewards

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

        opt_dynamics, opt_r = self.CalcPolicyData(policy)
        opt_V = np.linalg.inv(np.eye(self.n) - self.gamma * opt_dynamics) @ opt_r
        return policy, opt_V

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

    def CalcOptExpectedReward(self):
        return self.init_prob @ self.V
        # if 'offline' in params['eval_type']:
        #     return self.init_prob @ self.V
        # if 'online' in params['eval_type']:
        # expected_reward_vec = [self.opt_r @ (self.init_prob @ np.linalg.matrix_power(self.opt_P, i))
        #                        for i in range(params['steps'])]
        # batch_reward = [sum(group)
        #                 for group in np.array_split(expected_reward_vec, params['steps'] / params['eval_freq'])]
        # return np.cumsum(self.init_prob @ self.V)

        # raise ValueError('unexpected evaluation type')


class TreeMDP(MDPModel):
    def __init__(self, n, actions, chain_num, gamma, succ_num, resets_num=0, traps_num=0,
                 init_states_idx=frozenset({0}), **kwargs):

        self.n = n

        self.traps_idx = random.sample(self.GetActiveChains(), traps_num)
        self.init_states_idx = init_states_idx
        self.reset_states_idx = self.GenResetStates(resets_num=resets_num)

        super().__init__(n, actions, chain_num, gamma, succ_num, **kwargs)

    def get_successors(self, state_idx, **kwargs):
        if state_idx in self.reset_states_idx:
            return self.init_states_idx
        return super().get_successors(state_idx)

    def GenResetStates(self, **kwargs):
        possible_resets = set(range(self.n)).difference(self.init_states_idx)
        return random.sample(possible_resets, kwargs['resets_num'])

    def GenInitialProbability(self):
        init_prob = np.zeros(self.n)
        for state in self.init_states_idx:
            init_prob[state] = 1

        return init_prob / sum(init_prob)


class CliffWalker(TreeMDP):
    def __init__(self, size, gamma, random_prob, **kwargs):
        self.random_prob = random_prob
        self.size = size
        self.n = size ** 2
        self.actions: int = 4
        self.illegal_moves = {'row': [(0, 2), (self.size - 1, 0)],
                              'col': [(0, 3), (self.size - 1, 1)]}
        super().__init__(self.n, self.actions, 1, gamma, 4, **kwargs)

    def FindChain(self, state_idx):
        return 0

    def GetRewardParams(self, state_idx):
        return {'gauss_params': ((1, 0), 0)}

    def IsStateActionRewarded(self, state, action):
        return state == self.n - self.size

    def GenResetStates(self, **kwargs):
        return set(filter(lambda x: x % self.size == 0, list(range(self.n)))).difference({0, self.size ** 2})

    def convert_action_to_diff(self, action):
        if action == 0:
            return 1
        if action == 1:
            return self.size
        if action == 2:
            return -1
        if action == 3:
            return -self.size

    def calc_next_state(self, state, action):
        row = int(state % self.size)
        col = int(state / self.size)

        if (row, action) in self.illegal_moves['row'] or (col, action) in self.illegal_moves['col']:
            return None

        return state + self.convert_action_to_diff(action)

    def get_successors(self, state_idx, **kwargs):
        if state_idx in self.reset_states_idx:
            return self.init_states_idx

        desired_action = kwargs['action']
        desired_state = self.calc_next_state(state_idx, desired_action)
        undesired_actions = list(set(range(self.actions)).difference({desired_action}))
        undesired_states = [self.calc_next_state(state_idx, action) for action in undesired_actions]

        return desired_state, set(filter(lambda state: state is not None, undesired_states))

    def gen_row_of_P(self, succesors, state_idx):
        if state_idx in self.reset_states_idx:
            row = np.array([random.random() if idx in succesors else 0 for idx in range(self.n)])
            return row / sum(row)
        p_vec = np.zeros(self.n)
        desired_state = succesors[0]
        undesired_states = succesors[1]

        if desired_state is not None:
            p_vec[desired_state] = 1 - self.random_prob
            p_vec[list(undesired_states)] = self.random_prob / len(undesired_states)
        else:
            p_vec[state_idx] = 1

        return p_vec


class SeperateChainsMDP(TreeMDP):
    def __init__(self, n, actions, succ_num, reward_param, gamma, chain_num, op_succ_num,
                 resets_num=0, traps_num=0, **kwargs):
        self.chain_num = chain_num

        n += (1 - n % self.chain_num)  # make sure sub_chains are even sized
        self.chain_size = int((n - 1) / self.chain_num)

        self.chains = None
        self.buildChains()
        self.active_chains = self.GetActiveChains()
        self.reward_params = reward_param
        self.op_succ_num = op_succ_num

        super().__init__(n, actions=actions, chain_num=self.chain_num, gamma=gamma, traps_num=traps_num,
                         succ_num=succ_num, resets_num=resets_num)
        self.type = 'chains'

    @property
    def active_chains_ratio(self):
        return len(self.active_chains) / self.chain_num

    def buildChains(self):
        self.chains = [set(range(1 + i * self.chain_size, (i + 1) * self.chain_size + 1))
                       for i in range(self.chain_num)]

    def GenPossibleSuccessors(self, **kwargs):
        forbidden_states = self.GenForbiddenStates()

        possible_per_chain = [chain.difference(forbidden_states) for chain in self.chains]
        return [self.GenPossibleSuccessorsPerState(self.FindChain(s), possible_per_chain, state_idx=s)
                for s in range(self.n)]

    def GenForbiddenStates(self):
        return self.init_states_idx

    def GenPossibleSuccessorsPerState(self, chain_num, possible_per_chain, **kwargs):
        if chain_num is None:
            return [set(opt) for opt in possible_per_chain]

        return set(random.sample(possible_per_chain[chain_num], self.op_succ_num))

    def get_successors(self, state_idx, **kwargs):
        if state_idx in self.init_states_idx:
            max_states = min([len(chain_states) for chain_states in self.possible_suc[state_idx]])
            return reduce(lambda a, b: a.union(b), [set(random.sample(chain_succ, max_states))
                                                    for chain_succ in self.possible_suc[state_idx]])
        return super().get_successors(state_idx)

    def IsStateActionRewarded(self, state_idx, action):
        return self.FindChain(state_idx) in self.active_chains

    def FindChain(self, state_idx):
        if state_idx in self.init_states_idx:
            return None
        for i in range(self.chain_num):
            if state_idx < 1 + (i + 1) * self.chain_size:
                return i

    def GetRewardParams(self, state_idx):
        chain = self.FindChain(state_idx)
        if chain is None:
            return None

        if state_idx in self.traps_idx:
            return self.reward_params['trap']

        return self.reward_params.get(chain)

    def gen_row_of_P(self, succesors, state_idx):
        if state_idx in self.init_states_idx:
            res = np.array([1 if state in succesors else 0 for state in range(self.n)])
            return res / sum(res)

        return super().gen_row_of_P(succesors, state_idx)

    def GetActiveChains(self):
        return {self.chain_num - 1}


def GetSuccessorsInLine(state_idx, line_idxs, action):
    if state_idx == line_idxs[-1] or action != 0:
        return {line_idxs[0]}

    return {line_idxs[state_idx - line_idxs[0] + 1]}


class ChainsTunnelMDP(SeperateChainsMDP):
    def __init__(self, n, actions, succ_num, reward_param, gamma, chain_num, op_succ_num, tunnel_indexes, traps_num=0):
        self.tunnel_indexes = tunnel_indexes
        super().__init__(n, actions, succ_num, reward_param, gamma, chain_num, op_succ_num, traps_num)

    def IsStateActionRewarded(self, state_idx, action):
        if state_idx == self.tunnel_indexes[-1]:
            return action == 0

        return super().IsStateActionRewarded(state_idx, action)

    def GenForbiddenStates(self):
        return super().GenForbiddenStates().union(set(self.tunnel_indexes[1:]))

    def get_successors(self, state_idx, **kwargs):
        if state_idx in self.tunnel_indexes[:-1]:
            return GetSuccessorsInLine(state_idx, self.tunnel_indexes, kwargs['action'])

        return super().get_successors(state_idx, **kwargs)

    def GenPossibleSuccessors(self, **kwargs):
        kwargs['forbidden_states'] = self.tunnel_indexes[1:]
        return super().GenPossibleSuccessors(**kwargs)

    def GetRewardParams(self, state_idx):
        if state_idx == self.tunnel_indexes[-1]:
            return self.reward_params['tunnel_end']
        if state_idx in self.tunnel_indexes:
            return self.reward_params['lead_to_tunnel']
        return super().GetRewardParams(state_idx)


class StarMDP(SeperateChainsMDP):
    def __init__(self, n, actions, succ_num, reward_param, gamma, chain_num, op_succ_num, **kwargs):
        super().__init__(n, actions, succ_num, reward_param, gamma, chain_num, op_succ_num, **kwargs)
        self.chain_num += 1

    @property
    def active_chains_ratio(self):
        return len(self.active_chains) / (self.chain_num - 1)

    def FindChain(self, state_idx):
        if state_idx in self.init_states_idx:
            return self.chain_num - 1
        return super().FindChain(state_idx)

    def IsStateActionRewarded(self, state_idx, action):
        return state_idx not in self.init_states_idx

    def GenResetStates(self, resets_num):
        return [max(chain) for chain in self.chains]

    def get_successors(self, state_idx, **kwargs):
        if state_idx in self.init_states_idx:
            return self.chains[kwargs['action']].difference(self.reset_states_idx)
        return super().get_successors(state_idx)

    def GetActiveChains(self):
        return MDPModel.GetActiveChains(self)


class GittinsMDP(StarMDP):
    def __init__(self, n, actions, succ_num, reward_param, gamma, chain_num, op_succ_num, terminal_probability,
                 **kwargs):
        self.terminal_probability = terminal_probability
        super().__init__(n, actions, succ_num, reward_param, gamma, chain_num, op_succ_num, resets_num=chain_num,
                         **kwargs)

    def gen_row_of_P(self, succesors, state_idx):
        if state_idx in self.init_states_idx.union(self.reset_states_idx):
            return super().gen_row_of_P(succesors, state_idx)

        reset_state = self.reset_states_idx[self.FindChain(state_idx)]
        row = np.array([random.random() if idx in succesors.difference({reset_state}) else 0 for idx in range(self.n)])
        row *= ((1 - self.terminal_probability) / sum(row))

        row[reset_state] = self.terminal_probability
        return row

    def GetRewardParams(self, state_idx):
        chain = self.FindChain(state_idx)
        if chain is None:
            return None
        if state_idx in self.traps_idx:
            return self.reward_params['trap']

        reward_params = self.reward_params.get(chain)
        return reward_params['first'] if state_idx == min(self.chains[chain]) else reward_params['others']


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
    def get_successors(self, state_idx, **kwargs):
        return {np.mod(state_idx + 1, self.n)}


class SingleLineMDP(MDPModel):
    def IsStateActionRewarded(self, state_idx, action):
        return state_idx == self.n - 2 and action == 0

    def get_successors(self, state_idx, **kwargs):
        if kwargs['action'] == 0:  # forward -->
            return {(state_idx + 1) % self.n}
        return {0}


class BridgedMDP(SeperateChainsMDP):
    def __init__(self, bridges_num, chain_num=2, **kwargs):
        self.bridges_num = bridges_num
        self.bridge_states = None
        super().__init__(chain_num=chain_num, **kwargs)
        self.type = 'bridge'

    def GenPossibleSuccessorsPerState(self, chain_num, possible_per_chain, **kwargs):
        new_chain = 1 - chain_num if kwargs['state_idx'] in self.bridge_states else chain_num
        return super().GenPossibleSuccessorsPerState(new_chain, possible_per_chain)

    def IsStateActionRewarded(self, state_idx, act):
        return False if state_idx in self.bridge_states else super().IsStateActionRewarded(state_idx, act)

    def GetRewardParams(self, state_idx):
        if self.FindChain(state_idx) == 0:
            return self.reward_params['medium']

        return self.reward_params['high'] if np.random.random() < 0.5 else self.reward_params['low']

    def buildChains(self, **kwargs):
        super().buildChains()
        self.bridge_states = reduce(lambda a, b: a + b, [random.sample(chain, self.bridges_num)
                                                         for chain_num, chain in enumerate(self.chains)
                                                         if chain_num != kwargs.get('no_bridges_tunnels')])

    def GetActiveChains(self):
        return {0, 1}


class LevelsMDP(BridgedMDP):
    def __init__(self, bridges_num, chain_num, **kwargs):
        super().__init__(bridges_num, chain_num=chain_num, **kwargs)

    def GenPossibleSuccessorsPerState(self, chain_num, possible_per_chain, **kwargs):
        new_chain = 1 + chain_num if kwargs['state_idx'] in self.bridge_states else chain_num
        return SeperateChainsMDP.GenPossibleSuccessorsPerState(self, new_chain, possible_per_chain)

    def buildChains(self):
        super().buildChains(no_bridges_tunnels=self.chain_num - 1)

    def GetActiveChains(self):
        return MDPModel.GetActiveChains(self)

    def GetRewardParams(self, state_idx):
        return SeperateChainsMDP.GetRewardParams(self, state_idx)


if __name__ == "__main__":
    cliff = CliffWalker(5, 0.2, 0.9)
    print('doen')
