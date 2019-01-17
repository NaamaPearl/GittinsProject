import numpy as np
from Framework.PrioritizedObject import PrioritizedObject
from MDPModel.MDPBasics import StateScore
import random
from abc import abstractmethod

epsilon = 10 ** -5


class Prioritizer:
    def __init__(self, states, **kwargs):
        self.states = states

    def GradeStates(self):
        return {state.idx: random.random() for state in self.states}


class FunctionalPrioritizer(Prioritizer):
    def __init__(self, states, policy, p, r, temporal_extension, discount_factor):
        super().__init__(states)
        self.n = len(states)
        self.policy = policy
        self.temporal_extension = temporal_extension
        self.P = self.buildP(p, policy, temporal_extension)
        self.r = self.buildRewardVec(r, policy, temporal_extension, discount_factor)

    def buildP(self, p, policy, temporal_extension):
        prob_mat = [np.zeros((self.n, self.n)) for _ in range(temporal_extension)]
        for state in range(self.n):
            prob_mat[0][state] = p[state][policy[state]]

        for i in range(1, temporal_extension):
            prob_mat[i] = prob_mat[i - 1] @ prob_mat[0]

        return prob_mat

    def buildRewardVec(self, reward_mat, policy, temporal_extension, gamma):
        immediate_r = np.zeros(self.n)
        r = np.zeros(self.n)
        for idx in range(self.n):
            immediate_r[idx] = reward_mat[idx][policy[idx]]

        for state_idx in range(self.n):
            r[state_idx] = immediate_r[state_idx]
            for i in range(1, temporal_extension):
                p = self.P[i][state_idx]
                r[state_idx] += (gamma * (p @ immediate_r))

        return r

    @abstractmethod
    def GradeStates(self):
        pass


class GreedyPrioritizer(FunctionalPrioritizer):
    def GradeStates(self):
        return {state.idx: -self.r[state.idx] for state in self.states}


class GittinsPrioritizer(FunctionalPrioritizer):
    def GradeStates(self):
        """
        Identifies optimal state (maximal priority), updates result dictionary, and omits state from model.
        Operates Iteratively, until all states are ordered.
        """

        rs_list = [PrioritizedObject(s, StateScore(s, r)) for s, r in zip(self.states, self.r)]
        result = {}
        score = 1  # score is order of extraction

        while len(rs_list) > 1:
            # identify optimal state, omit it from model and add it to result
            opt_state = max(rs_list)
            rs_list.remove(opt_state)
            result[opt_state.idx] = score
            score += 1

            for rewarded_state in rs_list:
                self.CalcIndex(rewarded_state.reward, opt_state.reward)  # calc indexes after omission

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

    def CalcIndex(self, state: StateScore, opt_s: StateScore):
        """
        calc state's index after omission
        """
        P = self.P[-1]
        state_idx = state.idx
        opt_state_idx = opt_s.idx

        # calculate needed sizes for final calculations
        p_sub_optimal = 1 - P[state_idx, opt_state_idx]
        p_opt_stay = P[opt_state_idx, opt_state_idx] + epsilon
        sum_p_opt = 1 / (1 - p_opt_stay)
        t_opt_expect = 1 / (2 * (1 - p_opt_stay) ** 2)
        p_opt_back = P[state_idx, opt_state_idx] * sum_p_opt * (np.sum(P[opt_state_idx, :]) - p_opt_stay)

        R = p_sub_optimal * state.score + p_opt_back * (state.score + opt_s.score + opt_s.score * t_opt_expect)
        W = p_sub_optimal + p_opt_back * 2 + p_opt_back * t_opt_expect

        state.score = R / W
