import random
from abc import abstractmethod
import numpy as np
from Framework.PrioritizedObject import PrioritizedObject
from MDPModel.MDPBasics import StateScore
from MDPModel.MDPModel import MDPModel

epsilon = 10 ** -5


class Prioritizer:
    def __init__(self, states, **kwargs):
        self.n = len(states)
        self.states = states

    def grade_states(self):
        return self.build_result([(random.random(), i) for i in range(self.n)])

    @staticmethod
    def build_result(score_vec):
        score_vec.sort(reverse=True)
        return {state[1]: (order + 1, state[0]) for order, state in enumerate(score_vec)}


class FunctionalPrioritizer(Prioritizer):
    def __init__(self, states, policy, discount_factor, **kwargs):
        super().__init__(states)
        self.policy = policy
        self.discount_factor = discount_factor

    @abstractmethod
    def grade_states(self):
        pass


class TabularPrioritizer(FunctionalPrioritizer):
    def __init__(self, states, policy, p, r, temporal_extension, discount_factor, **kwargs):
        super().__init__(states, policy, discount_factor)
        self.temporal_extension = temporal_extension

        def build_p():
            prob_mat = [np.zeros((self.n, self.n)) for _ in range(self.temporal_extension)]
            for state in range(self.n):
                prob_mat[0][state] = p[state][self.policy[state]]

            for i in range(1, self.temporal_extension):
                prob_mat[i] = prob_mat[i - 1] @ prob_mat[0]

            return prob_mat

        def build_reward_vec():
            immediate_r = np.zeros(self.n)
            new_r = np.zeros(self.n)
            for idx in range(self.n):
                try:
                    immediate_r[idx] = r[idx][self.policy[idx]]
                except IndexError:
                    immediate_r[idx] = r[idx]

            for state_idx in range(self.n):
                new_r[state_idx] = immediate_r[state_idx]
                for i in range(1, self.temporal_extension):
                    state_p = self.P[i][state_idx]
                    new_r[state_idx] += (self.discount_factor * (state_p @ immediate_r))

            return new_r

        self.P = build_p()
        self.r = build_reward_vec()

    @abstractmethod
    def grade_states(self):
        pass


class GreedyPrioritizer(TabularPrioritizer):
    def grade_states(self):
        return self.build_result([(v, i) for i, v in (enumerate(list(self.r)))])


class GittinsPrioritizer(TabularPrioritizer):
    def grade_states(self):
        """
        Identifies optimal state (maximal priority), updates result dictionary, and omits state from model.
        Operates Iteratively, until all states are ordered.
        """

        def update_prob_mat():
            """ Calculate new transition probabilities, after optimal state omission (invoked after removal) """
            P = self.P[-1]
            for state1 in rs_list:
                s1 = state1.object.idx
                if P[s1, opt_state.idx] > 0:  # only update P for states from which opt_s is reachable
                    for state2 in rs_list:
                        s2 = state2.object.idx
                        P[s1, s2] += ((P[s1, opt_state.idx] * P[opt_state.idx, s2]) /
                                      (1 + epsilon - P[opt_state.idx, opt_state.idx]))

            # zero out transitions to/ from opt_state
            P[opt_state.idx] = 0
            P[:, opt_state.idx] = 0

        def update_state_index(state: StateScore):
            """ Calc state's index after omission """
            P = self.P[-1]
            state_idx = state.idx
            opt_state_idx = opt_state.reward.idx

            # calculate needed sizes for final calculations
            p_opt_stay = P[opt_state_idx, opt_state_idx] + epsilon
            p_opt_sum = 1 / (1 - p_opt_stay)
            p_s_not_opt = 1 - P[state_idx, opt_state_idx]
            p_s_opt = P[state_idx, opt_state_idx]
            p_opt_not_opt = (np.sum(P[opt_state_idx, :]) - p_opt_stay)
            p_s_opt_not_opt = p_s_opt * p_opt_sum * p_opt_not_opt

            R = state.score * (p_s_not_opt + p_s_opt_not_opt) + \
                (gamma * p_s_opt * p_opt_not_opt * opt_state.reward.score / (1 - gamma)) * \
                (p_opt_sum - gamma / (1 - p_opt_stay * gamma))
            W = p_s_not_opt + \
                (p_s_opt * p_opt_not_opt / (1 - gamma)) * \
                (p_opt_sum - gamma ** 2 / (1 - p_opt_stay * gamma))
            state.score = R / W

        gamma = self.discount_factor
        rs_list = [PrioritizedObject(s, StateScore(s, r)) for s, r in zip(self.states, self.r)]
        result = {}
        order = 1  # score is order of extraction

        while len(rs_list) > 1:
            # identify optimal state, omit it from model and add it to result
            opt_state = max(rs_list)
            rs_list.remove(opt_state)
            result[opt_state.idx] = (order, opt_state.reward.score)
            order += 1

            [update_state_index(rewarded_state.reward) for rewarded_state in rs_list]  # calc indexes after omission
            update_prob_mat()  # calc new transition matrix

        # when only one state remains, simply add it to the result list
        last_state = rs_list.pop()
        result[last_state.object.idx] = (order, last_state.reward.score)
        return result


class ModelFreeGittinsPrioritizer(FunctionalPrioritizer):
    def __init__(self, states, policy, discount_factor, p, trajectory_num, max_trajectory_len, parameter, r, **kwargs):
        super().__init__(states, policy, discount_factor)
        self.model: MDPModel = p
        self.trajectory_num = trajectory_num
        self.max_trajectory_len = max_trajectory_len
        self.parameter = parameter
        self.reward = r

    def grade_states(self):
        def calc_state_index(state_idx):
            def create_trajectory():
                res = []
                curr_state = state_idx

                for i in range(self.max_trajectory_len):
                    next_state, new_reward = self.get_state_sim_result(curr_state)

                    res.append(new_reward * (self.discount_factor ** i))
                    curr_state = next_state

                return np.cumsum(np.array(res)) / denom_vec

            trajectory_mat = np.empty((self.trajectory_num, self.max_trajectory_len))
            for trajectory_num in range(self.trajectory_num):
                trajectory_mat[trajectory_num] = create_trajectory()

            return max(np.mean(trajectory_mat, axis=0))

        denom_vec = np.array([(self.discount_factor ** i - 1) / (self.discount_factor - 1)
                              for i in range(1, self.max_trajectory_len + 1)])

        return self.build_result([(calc_state_index(state_idx), state_idx) for state_idx in range(self.n)])

    def get_state_sim_result(self, state):
        """Simulate one step, and retrieve next_state and reward. If prioritization is done according to error, return
        estimated error instead of error"""
        next_state, reward = self.model.sample_state_action(state, self.policy[state])
        reward = reward if self.parameter == 'reward' else abs(self.reward[next_state][self.policy[next_state]])

        return next_state, reward
