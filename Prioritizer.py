import numpy as np
import random
from framework import PrioritizedObject


epsilon = 10 ** -5


class Prioritizer:
    def __init__(self):
        self.n = None
        self.policy = None
        self.r = None
        self.P = None

    def GradeStates(self, **kwargs):
        return {state.idx: random.random() for state in kwargs['states']}


class GreedyPrioritizer(Prioritizer):
    def GradeStates(self, **kwargs):
        return {s: -max(abs(kwargs['r'][s])) for s in range(len(kwargs['states']))}


class GittinsPrioritizer(Prioritizer):
    def InitProbMat(self, p, look_ahead):
        prob_mat = [np.zeros((self.n, self.n)) for _ in range(look_ahead)]
        for state in range(self.n):
            action = self.policy[state]
            prob_mat[0][state] = p[state][action]

        for i in range(1, look_ahead):
            prob_mat[i] = prob_mat[i - 1] @ prob_mat[0]

        return prob_mat

    def InitRewardVec(self, reward_mat, look_ahead, gamma):
        immediate_r = np.zeros(self.n)
        r = np.zeros(self.n)
        for idx in range(self.n):
            immediate_r[idx] = reward_mat[idx][self.policy[idx]]

        for state_idx in range(self.n):
            r[state_idx] = immediate_r[state_idx]
            for i in range(1, look_ahead):
                p = self.P[i][state_idx]
                r[state_idx] += (gamma * (p @ immediate_r))

        return r

    def GradeStates(self, **kwargs):
        """
        Identifies optimal state (maximal priority), updates result dictionary, and omits state from model.
        Operates Iteratively, until all states are ordered.
        """

        if kwargs['random_prio']:
            return super().GradeStates(**kwargs)

        self.n = len(kwargs['states'])
        self.policy = kwargs['policy']
        self.P = self.InitProbMat(kwargs['p'], kwargs['look_ahead'])
        r = self.InitRewardVec(kwargs['r'], kwargs['look_ahead'], kwargs['discount'])

        rs_list = [PrioritizedObject(s, r) for s, r in zip(kwargs['states'], r)]
        result = {}
        score = 1  # score is order of extraction

        while len(rs_list) > 1:
            # identify optimal state, omit it from model and add it to result
            opt_state = max(rs_list)
            rs_list.remove(opt_state)
            result[opt_state.idx] = score
            score += 1

            for rewarded_state in rs_list:
                self.CalcIndex(rewarded_state, opt_state)  # calc index after omission, for all remaining states

            self.CalcNewProb(rs_list, opt_state)  # calc new transition matrix
        last_state = rs_list.pop()
        result[last_state.object.idx] = score  # when only one state remains, simply add it to the result list
        return result

    def CalcNewProb(self, rs_list, opt_s: PrioritizedObject):
        """
    calculate new transition probabilities, after optimal state omission (invoked after removal)
    """
        P = self.P[-1]
        for state1 in rs_list:
            s1 = state1.object.idx
            if P[s1, opt_s.idx] > 0:  # only update P for states from which opt_s is reachable
                for state2 in rs_list:
                    s2 = state2.object.idx
                    P[s1, s2] += (P[s1, opt_s.idx] * P[opt_s.idx, s2] / (1 + epsilon - P[opt_s.idx, opt_s.idx]))

        # zero out transitions to/ from opt_state
        P[opt_s.idx] = 0
        P[:, opt_s.idx] = 0

    def CalcIndex(self, state: PrioritizedObject, opt_s: PrioritizedObject):
        """
        calc state's index after omission
        """
        action = self.policy[opt_s.idx]
        P = self.P[-1]
        state_idx = state.idx
        opt_state_idx = opt_s.idx

        # in case we haven't visited this state yet
        if np.sum(P[action]) == 0:  # TODO: T bored
            return 100

        # calculate needed sizes for final calculations
        p_sub_optimal = 1 - P[state_idx, opt_state_idx]
        p_opt_stay = P[opt_state_idx, opt_state_idx] + epsilon
        sum_p_opt = 1 / (1 - p_opt_stay)
        t_opt_expect = 1 / (2 * (1 - p_opt_stay) ** 2)
        p_opt_back = P[state_idx, opt_state_idx] * sum_p_opt * (np.sum(P[opt_state_idx, :]) - p_opt_stay)

        R = p_sub_optimal * state.reward + p_opt_back * (state.reward + opt_s.reward + opt_s.reward * t_opt_expect)
        W = p_sub_optimal + p_opt_back * 2 + p_opt_back * t_opt_expect

        state.reward = R / W