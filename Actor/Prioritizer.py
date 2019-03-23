import random
from abc import abstractmethod

import numpy as np

from Framework.PrioritizedObject import PrioritizedObject
from MDPModel.MDPBasics import StateScore
from MDPModel.MDPModel import MDPModel

epsilon = 10 ** -5


class Prioritizer:
    def __init__(self, states, **kwargs):
        self.states = states

    def GradeStates(self):
        return {state.idx: random.random() for state in self.states}, None


class FunctionalPrioritizer(Prioritizer):
    def __init__(self, states, policy, discount_factor, **kwargs):
        super().__init__(states)
        self.n = len(states)
        self.policy = policy
        self.discount_factor = discount_factor

    @abstractmethod
    def GradeStates(self):
        pass


class TabularPrioritizer(FunctionalPrioritizer):
    def __init__(self, states, policy, p, r, temporal_extension, discount_factor, **kwargs):
        super().__init__(states, policy, discount_factor)
        self.temporal_extension = temporal_extension

        def buildP():
            prob_mat = [np.zeros((self.n, self.n)) for _ in range(self.temporal_extension)]
            for state in range(self.n):
                prob_mat[0][state] = p[state][self.policy[state]]

            for i in range(1, self.temporal_extension):
                prob_mat[i] = prob_mat[i - 1] @ prob_mat[0]

            return prob_mat

        def buildRewardVec():
            immediate_r = np.zeros(self.n)
            new_r = np.zeros(self.n)
            for idx in range(self.n):
                immediate_r[idx] = r[idx][self.policy[idx]]

            for state_idx in range(self.n):
                new_r[state_idx] = immediate_r[state_idx]
                for i in range(1, self.temporal_extension):
                    state_p = self.P[i][state_idx]
                    new_r[state_idx] += (self.discount_factor * (state_p @ immediate_r))

            return new_r

        self.P = buildP()
        self.r = buildRewardVec()

    @abstractmethod
    def GradeStates(self):
        pass


class GreedyPrioritizer(TabularPrioritizer):
    def GradeStates(self):
        return {state.idx: -self.r[state.idx] for state in self.states}


class GittinsPrioritizer(TabularPrioritizer):
    def GradeStates(self):
        """
        Identifies optimal state (maximal priority), updates result dictionary, and omits state from model.
        Operates Iteratively, until all states are ordered.
        """

        def UpdateProbMat():
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

        def UpdateStateIndex(state: StateScore):
            """ Calc state's index after omission """
            P = self.P[-1]
            state_idx = state.idx
            opt_state_idx = opt_state.reward.idx

            # calculate needed sizes for final calculations
            p_sub_optimal = 1 - P[state_idx, opt_state_idx]
            p_opt_stay = P[opt_state_idx, opt_state_idx] + epsilon
            sum_p_opt = 1 / (1 - p_opt_stay)
            t_opt_expect = 1 / (2 * (1 - p_opt_stay) ** 2)
            p_opt_back = P[state_idx, opt_state_idx] * sum_p_opt * (np.sum(P[opt_state_idx, :]) - p_opt_stay)

            R = p_sub_optimal * state.score + p_opt_back * \
                (state.score + opt_state.reward.score + opt_state.reward.score * t_opt_expect)
            W = p_sub_optimal + p_opt_back * 2 + p_opt_back * t_opt_expect

            state.score = R / W

        rs_list = [PrioritizedObject(s, StateScore(s, r)) for s, r in zip(self.states, self.r)]
        result = {}
        order = 1  # score is order of extraction

        while len(rs_list) > 1:
            # identify optimal state, omit it from model and add it to result
            opt_state = max(rs_list)
            rs_list.remove(opt_state)
            result[opt_state.idx] = (order, opt_state.reward.score)
            order += 1

            [UpdateStateIndex(rewarded_state.reward) for rewarded_state in rs_list]  # calc indexes after omission
            UpdateProbMat()  # calc new transition matrix

        # when only one state remains, simply add it to the result list
        last_state = rs_list.pop()
        result[last_state.object.idx] = (order, last_state.reward.score)
        return result


class ModelFreeGittinsPrioritizer(FunctionalPrioritizer):
    def __init__(self, states, policy, discount_factor, p, trajectory_num, max_trajectory_len, **kwargs):
        super().__init__(states, policy, discount_factor)
        self.model: MDPModel = p
        self.trajectory_num = trajectory_num
        self.max_trajectory_len = max_trajectory_len

    def GradeStates(self):
        def CalcStateIndex(state_idx):
            def create_trajectory():
                res = [0.]
                curr_state = state_idx

                for i in range(self.max_trajectory_len):
                    next_state, new_reward = self.model.sample_state_action(curr_state, self.policy[curr_state])

                    res.append(res[-1] + self.discount_factor ** i * new_reward)
                    curr_state = next_state

                return np.array(res[1:]) / denom_vec

            trajectory_mat = np.empty((self.trajectory_num, self.max_trajectory_len))
            for trajectory_num in range(self.trajectory_num):
                trajectory_mat[trajectory_num] = create_trajectory()

            return max(np.mean(trajectory_mat, axis=0))

        denom_vec = np.array([(self.discount_factor ** i - 1) / (self.discount_factor - 1)
                              for i in range(1, self.max_trajectory_len + 1)])

        sorted_state_list = sorted({state_idx: -CalcStateIndex(state_idx) for state_idx in range(self.n)}.items(),
                                   key=lambda x: x[1])
        return {state[0]: (order + 1, state[1]) for order, state in enumerate(sorted_state_list)}


class ModelFreeGittinsPrioritizer_obselete(FunctionalPrioritizer):
    def __init__(self, states, policy, p, discount_factor, **kwargs):
        super().__init__(states, policy, discount_factor)
        self.model: MDPModel = p
        self.max_trajectory_len = 2
        self.trajectories_per_len = 2

    def GradeStates(self):
        def CalcStateIndex(state_idx):
            def MaxTrajectoryPerLength():
                def CalcTrajectoryValue():
                    accumulated_reward = 0
                    curr_state: int = state_idx

                    for i in range(trajectory_len):
                        policy_action = self.policy[curr_state]
                        next_state, new_reward = self.model.sample_state_action(curr_state, policy_action)

                        curr_state = next_state
                        accumulated_reward += (self.discount_factor ** i * new_reward)

                    return accumulated_reward

                expected_maximal_trajectory_value = 0
                for _ in range(self.trajectories_per_len):
                    expected_maximal_trajectory_value += CalcTrajectoryValue()

                return expected_maximal_trajectory_value / self.trajectories_per_len

            curr_max = -np.inf
            for trajectory_len in range(self.max_trajectory_len):
                curr_max = max(curr_max, MaxTrajectoryPerLength())

            return curr_max

        sorted_state_list = sorted({state_idx: -CalcStateIndex(state_idx) for state_idx in range(self.n)}.items(),
                                   key=lambda x: x[1])
        return {state[0]: (order + 1, state[1]) for order, state in enumerate(sorted_state_list)}
