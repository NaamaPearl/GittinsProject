import random
from functools import reduce
from itertools import product

from MDPModel.RewardGenerator import *

threshold = 10 ** -10


class MDPModel:
    def __init__(self, n, actions, chain_num, gamma, succ_num, **kwargs):
        self.n: int = n
        self.type = 'regular'
        self.chain_num = chain_num
        self.actions: int = actions
        self.init_prob = self.gen_initial_probability()

        self.succ_num = succ_num
        self.possible_suc = None
        self.P = self.build_p()

        self.r = self.gen_r_mat()
        self.expected_r = np.array(
            [[self.get_expected_reward(s, a) for s in range(self.n)] for a in range(self.actions)])
        self.gamma = gamma
        self.opt_policy, self.V = self.calc_opt_policy()

    def get_expected_reward(self, s, a):
        return self.r.get((s, a), RewardGenerator()).expected_reward

    def get_next_state(self, state_idx, action):
        return np.random.choice(range(self.n), p=self.P[state_idx][action])

    def sample_state_action(self, state_idx, action):
        return self.get_next_state(state_idx, action), self.generate_reward(state_idx, action)

    def generate_reward(self, state_idx, action):
        return self.r.get((state_idx, action), RewardGenerator()).GiveReward()

    @property
    def active_chains_ratio(self):
        return 1

    def build_p(self):
        self.possible_suc = self.gen_possible_successors()

        return [np.array([self.gen_p_matrix(state_idx, self.get_successors(state_idx, action=act))
                          for act in range(self.actions)]) for state_idx in range(self.n)]

    def gen_possible_successors(self, **kwargs):
        all_states = list(range(self.n))
        return {s: all_states for s in all_states}

    def get_active_chains(self):
        return {0}

    def is_s_a_rewarded(self, state, action):
        return True

    def find_chain(self, state_idx):
        return 0

    def get_successors(self, state_idx, **kwargs):
        return set(random.sample(self.possible_suc[state_idx], self.succ_num))

    def gen_p_matrix(self, state_idx, successors):
        if self.is_sink_state(state_idx):
            self_vec = np.zeros(self.n)
            self_vec[state_idx] = 1
            return np.array(self_vec)
        else:
            return np.array(self.gen_row_of_p(successors, state_idx))

    def gen_row_of_p(self, successors, state_idx):

        row = np.array([random.random() if idx in successors else 0 for idx in range(self.n)])
        return row / sum(row)

    def get_reward_params(self, state_idx):
        return None

    def gen_r_mat(self):
        active_sa = filter(lambda sa: self.is_s_a_rewarded(*sa), product(range(self.n), range(self.actions)))
        res = {(s, a): RewardGeneratorFactory.generate(reward_params=self.get_reward_params(s)) for (s, a) in active_sa}

        return res

    @staticmethod
    def is_sink_state(*args):
        return False

    def gen_initial_probability(self):
        return np.ones(self.n) / self.n

    def calc_policy_data(self, policy):
        policy_dynamics = np.zeros((self.n, self.n))
        policy_expected_rewards = np.zeros(self.n)
        for s, a in enumerate(policy):
            policy_dynamics[s] = self.P[s][a]
            policy_expected_rewards[s] = self.get_expected_reward(s, a)

        return policy_dynamics, policy_expected_rewards

    def calc_opt_policy(self):
        v = np.zeros(self.n)
        v_old = np.ones(self.n)
        policy = np.zeros(self.n, dtype=int)
        while any(abs(v - v_old) > threshold):
            v_old = np.copy(v)
            for s in range(self.n):
                r = [self.get_expected_reward(s, a) for a in range(self.actions)]
                v_new = r + (self.P[s] @ (self.gamma * v_old))
                v[s] = max(v_new)
                policy[s] = np.argmax(v_new)

        opt_dynamics, opt_r = self.calc_policy_data(policy)
        opt_v = np.linalg.inv(np.eye(self.n) - self.gamma * opt_dynamics) @ opt_r
        return policy, opt_v

    @property
    def opt_r(self):
        return np.array([self.get_expected_reward(s, self.opt_policy[s]) for s in range(self.n)])

    @property
    def opt_p(self):
        prob_mat = np.zeros((self.n, self.n))
        for state in range(self.n):
            action = self.opt_policy[state]
            prob_mat[state] = self.P[state][action]
        return prob_mat

    def calc_opt_expected_reward(self):
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
    """ Introducing traps, initial states and reset states to the main MDP class"""

    def __init__(self, n, actions, chain_num, gamma, succ_num, resets_num=0, traps_num=0,
                 init_states_idx=frozenset({0}), **kwargs):

        self.n = n

        self.traps_idx = random.sample(self.get_active_chains(), traps_num)
        self.init_states_idx = init_states_idx
        self.reset_states_idx = self.gen_reset_states(resets_num=resets_num)

        super().__init__(n, actions, chain_num, gamma, succ_num, **kwargs)

    def get_successors(self, state_idx, **kwargs):
        if state_idx in self.reset_states_idx:
            return self.init_states_idx
        return super().get_successors(state_idx)

    def gen_reset_states(self, **kwargs):
        possible_resets = set(range(self.n)).difference(self.init_states_idx)
        return random.sample(possible_resets, kwargs['resets_num'])

    def gen_initial_probability(self):
        init_prob = np.zeros(self.n)
        for state in self.init_states_idx:
            init_prob[state] = 1

        return init_prob / sum(init_prob)


class DirectedTreeMDP(TreeMDP):
    """ Directional graph. State can only lead to deeper states (except leaves which lead back to the root"""

    def __init__(self, depth, actions, gamma, resets_num, **kwargs):
        def get_level_states(level_depth):
            return 2 ** level_depth - 1, 2 ** (level_depth + 1) - 1

        self.tree_depth = depth
        self.levels = list(map(get_level_states, range(self.tree_depth)))
        self.reward_param = {'basic': {'gauss_params': ((0, 1), 0)},
                             17: {'gauss_params': ((50, 0), 0)}, 25: {'gauss_params': ((-50, 0), 0)}}
        super().__init__(n=2 ** depth - 1, actions=actions, chain_num=1, gamma=gamma, succ_num=2, resets_num=resets_num,
                         **kwargs)

    def get_reward_params(self, state_idx):
        return self.reward_param.get(state_idx, self.reward_param['basic'])

    def gen_possible_successors(self, **kwargs):
        """ each state can lead to states one level deeper, excepts leaves, which lead to the initial states"""

        def calc_depth(state_idx):
            return int(np.log2(state_idx + 1))

        res = [self.levels[calc_depth(state_idx) + 1]
               for state_idx in list(reduce(lambda a, b: a.union(b), self.levels[:-1]))]
        res += [self.init_states_idx for _ in self.leafs]

        return res

    def gen_reset_states(self, **kwargs):
        """ reset states are evenly distributed amongs tree's inner nodes """
        possible_resets = set(reduce(lambda a, b: a.union(b), self.levels[1:-1]))
        return random.sample(possible_resets, kwargs['resets_num']) + list(self.leafs)

    @property
    def leafs(self):
        return self.levels[-1]


class CliffWalker(TreeMDP):
    """ In this MDP, state-space is a rectangle. It's bottom edge is considered a cliff edge, from which an agent might
     fall. In such an occasion, the falling agent will be returned to the initial state.
     Initial state is left-bottom corner, and reward is only present at the bottom-right corner."""

    def __init__(self, size, gamma, random_prob, **kwargs):
        self.random_prob = random_prob
        self.size = size
        self.n = size ** 2
        self.actions: int = 4  # up, down, right, left
        self.illegal_moves = {'row': [(0, 2), (self.size - 1, 0)],
                              'col': [(0, 3), (self.size - 1, 1)]}  # edge states
        super().__init__(self.n, self.actions, 1, gamma, 4, **kwargs)

    def get_reward_params(self, state_idx):
        return {'gauss_params': ((1, 0), 0)}

    def is_s_a_rewarded(self, state, action):
        """ only right-bottom corner"""
        return state == self.n - self.size

    def gen_reset_states(self, **kwargs):
        """ all bottom states, except the initial state"""
        return set(filter(lambda x: x % self.size == 0, range(self.n))).difference(self.init_states_idx)

    def get_successors(self, state_idx, **kwargs):
        if state_idx in self.reset_states_idx:
            return self.init_states_idx

        def calc_next_state(action):
            row = int(state_idx % self.size)
            col = int(state_idx / self.size)

            if (row, action) in self.illegal_moves['row'] or (col, action) in self.illegal_moves['col']:
                return

            def convert_action_to_diff():
                if action == 0:
                    return 1
                if action == 1:
                    return self.size
                if action == 2:
                    return -1
                if action == 3:
                    return -self.size

            return state_idx + convert_action_to_diff()

        desired_action = kwargs['action']
        desired_state = calc_next_state(desired_action)

        undesired_actions = set(range(self.actions)) - {desired_action}
        undesired_states = set(filter(lambda state: state is not None, map(calc_next_state, undesired_actions)))

        return desired_state, undesired_states

    def gen_row_of_p(self, successors, state_idx):
        if state_idx in self.reset_states_idx:
            row = np.array([random.random() if idx in successors else 0 for idx in range(self.n)])
            return row / sum(row)

        p_vec = np.zeros(self.n)
        desired_state = successors[0]
        undesired_states = successors[1]

        # if action is legal, give it's state 1 - random_prob, while other possible share the remaining
        if desired_state is not None:
            p_vec[desired_state] = 1 - self.random_prob
            p_vec[list(undesired_states)] = self.random_prob / len(undesired_states)
        else:
            p_vec[state_idx] = 1

        return p_vec


class CliquesMDP(TreeMDP):
    """ Initial state leads to different isolated cliques, each with different reward parameters"""

    def __init__(self, n, actions, succ_num, reward_param, gamma, chain_num, op_succ_num,
                 resets_num=0, traps_num=0, **kwargs):
        self.chain_num = chain_num

        n += (1 - n % self.chain_num)  # make sure sub_chains are even sized
        self.chain_size = int((n - 1) / self.chain_num)

        self.chains = None
        self.build_chains()
        self.active_chains = self.get_active_chains()
        self.reward_params = reward_param
        self.op_succ_num = op_succ_num

        super().__init__(n, actions=actions, chain_num=self.chain_num, gamma=gamma, traps_num=traps_num,
                         succ_num=succ_num, resets_num=resets_num)
        self.type = 'chains'

    @property
    def active_chains_ratio(self):
        return len(self.active_chains) / self.chain_num

    def build_chains(self):
        self.chains = [set(range(1 + i * self.chain_size, (i + 1) * self.chain_size + 1))
                       for i in range(self.chain_num)]

    def gen_possible_successors(self, **kwargs):
        forbidden_states = self.gen_forbidden_states()

        # reachable states from every chain
        possible_per_chain = [chain.difference(forbidden_states) for chain in self.chains]
        return {s: self.gen_possible_successors_per_state(self.find_chain(s), possible_per_chain)
                for s in range(self.n)}

    def gen_forbidden_states(self):
        return self.init_states_idx

    def gen_possible_successors_per_state(self, chain_num, possible_per_chain):
        """ Initial states lead to all reachable staets (per chain). Any other state only lead to a small group of
        states"""

        if chain_num is None:
            return {chain: set(opt) for chain, opt in enumerate(possible_per_chain)}

        return set(random.sample(possible_per_chain[chain_num], self.op_succ_num))

    def get_successors(self, state_idx, **kwargs):
        """ Initial state leads evenly to all chains. Other states only to a small group of states from their chain"""
        if state_idx not in self.init_states_idx:
            return super().get_successors(state_idx)

        def choose_possible_succ(chain_succ):
            return set(random.sample(chain_succ, max_states))

        max_states = min(map(lambda x: len(x), self.possible_suc[state_idx].values()))
        return reduce(lambda a, b: a.union(b), map(choose_possible_succ, self.possible_suc[state_idx].values()))

    def is_s_a_rewarded(self, state_idx, action):
        return self.find_chain(state_idx) in self.active_chains

    def find_chain(self, state_idx):
        if state_idx in self.init_states_idx:
            return None
        for i in range(self.chain_num):
            if state_idx < 1 + (i + 1) * self.chain_size:
                return i

    def get_reward_params(self, state_idx):
        chain = self.find_chain(state_idx)
        if chain is None:
            return None

        if state_idx in self.traps_idx:
            return self.reward_params['trap']

        return self.reward_params.get(chain)

    def gen_row_of_p(self, successors, state_idx):
        if state_idx in self.init_states_idx:
            res = np.array([1 if state in successors else 0 for state in range(self.n)])
            return res / sum(res)

        return super().gen_row_of_p(successors, state_idx)

    def get_active_chains(self):
        return {self.chain_num - 1}


class SeparateChainsMDP(CliquesMDP):
    pass


class ChainsTunnelMDP(CliquesMDP):
    def __init__(self, n, actions, succ_num, reward_param, gamma, chain_num, op_succ_num, tunnel_indexes, traps_num=0):
        self.tunnel_indexes = tunnel_indexes
        super().__init__(n, actions, succ_num, reward_param, gamma, chain_num, op_succ_num, traps_num)

    def is_s_a_rewarded(self, state_idx, action):
        if state_idx == self.tunnel_indexes[-1]:
            return action == 0

        return super().is_s_a_rewarded(state_idx, action)

    def gen_forbidden_states(self):
        """ Tunnel states are unreachable from states outside the tunnel"""
        return super().gen_forbidden_states().union(set(self.tunnel_indexes[1:]))

    def get_successors(self, state_idx, **kwargs):
        def get_successors_in_line():
            """ Action 0 leads one step forward. All other actions lead to line start"""
            if state_idx == self.tunnel_indexes[-1] or kwargs['action'] != 0:
                return {self.tunnel_indexes[0]}

            return {self.tunnel_indexes[state_idx - self.tunnel_indexes[0] + 1]}

        if state_idx in self.tunnel_indexes[:-1]:
            return get_successors_in_line()

        return super().get_successors(state_idx, **kwargs)

    def gen_possible_successors(self, **kwargs):
        kwargs['forbidden_states'] = self.tunnel_indexes[1:]
        return super().gen_possible_successors(**kwargs)

    def get_reward_params(self, state_idx):
        if state_idx == self.tunnel_indexes[-1]:
            return self.reward_params['tunnel_end']
        if state_idx in self.tunnel_indexes:
            return self.reward_params['lead_to_tunnel']
        return super().get_reward_params(state_idx)


class StarMDP(CliquesMDP):
    """ Separate MDPs, all connected via one initial state"""

    def __init__(self, n, actions, succ_num, reward_param, gamma, chain_num, op_succ_num, **kwargs):
        super().__init__(n, actions, succ_num, reward_param, gamma, chain_num, op_succ_num, **kwargs)
        self.chain_num += 1

    @property
    def active_chains_ratio(self):
        return len(self.active_chains) / (self.chain_num - 1)

    def find_chain(self, state_idx):
        if state_idx in self.init_states_idx:
            return self.chain_num - 1
        return super().find_chain(state_idx)

    def is_s_a_rewarded(self, state_idx, action):
        return state_idx not in self.init_states_idx

    def gen_reset_states(self, resets_num):
        return [max(chain) for chain in self.chains]

    def get_successors(self, state_idx, **kwargs):
        if state_idx in self.init_states_idx:
            return self.chains[kwargs['action']].difference(self.reset_states_idx)
        return super().get_successors(state_idx)

    def get_active_chains(self):
        return MDPModel.get_active_chains(self)
