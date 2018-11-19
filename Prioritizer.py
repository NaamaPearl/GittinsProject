from copy import copy
import numpy as np
import random
from MDPModel import PrioritizedObject, MDPModel

epsilon = 10 ** -10


class Prioritizer:
    def __init__(self, model: MDPModel):
        self.model = model

    def GradeStates(self):
        return {state_idx: random.random() for state_idx in range(self.model.n)}


class GittinsPrioritizer(Prioritizer):
    def __init__(self, model: MDPModel, approximation=True):
        super().__init__(model)
        if approximation:
            self.P = copy(model.P_hat)
            self.R = copy(model.r_hat)
        else:
            self.P = copy(model.P)
            self.R = copy(model.r)

    def GradeStates(self):
        """
        Identifies optimal state (maximal priority), updates result dictionary, and omits state from model.
        Operates Iteratively, until all states are ordered.
        """
        rs_list = [PrioritizedObject(state, self.R[i]) for i, state in enumerate(self.model.s)]
        result = {}
        score = 1  # score is order of extraction

        while len(rs_list) > 1:
            # identify optimal state, omit it from model and add it to result
            opt_state = max(rs_list)
            rs_list.remove(opt_state)
            result[opt_state.object.idx] = score
            score += 1

            for rewarded_state in rs_list:
                self.CalcIndex(rewarded_state, opt_state)  # calc index after omission, for all remaining states

            self.CalcNewProb(rs_list, opt_state.object.idx)  # calc new transition matrix

        result[rs_list.pop().object.idx] = score  # when only one state remains, simply add it to the result list
        return result

    def CalcNewProb(self, rs_list, opt_s):
        """
    calculate new transition probabilities, after optimal state omission (invoked after removal)
    """
        for state1 in rs_list:
            s1 = state1.object.idx
            if self.P[s1, opt_s] > 0:  # only update P for states from which opt_s is reachable
                for state2 in rs_list:
                    s2 = state2.object.idx
                    self.P[s1, s2] += (self.P[s1, opt_s] * self.P[opt_s, s2] / (1 + epsilon - self.P[opt_s, opt_s]))

        # zero out transitions to/ from opt_state
        self.P[opt_s] = 0
        self.P[:, opt_s] = 0

    def CalcIndex(self, state, opt_s):
        """
        calc state's index after omission
        """
        state_idx = state.object.idx
        opt_state_idx = opt_s.object.idx

        # in case we haven't visited this state yet
        if np.sum(self.P[state_idx, :]) == 0:  # TODO: T bored
            return 100

        # calculate needed sizes for final calculations
        p_sub_optimal = 1 - self.P[state_idx, opt_state_idx]
        p_opt_stay = self.P[opt_state_idx, opt_state_idx] + epsilon
        sum_p_opt = 1 / (1 - p_opt_stay)
        t_opt_expect = 1 / (2 * (1 - p_opt_stay) ** 2)
        p_opt_back = self.P[state_idx, opt_state_idx] * sum_p_opt * (np.sum(self.P[opt_state_idx, :]) - p_opt_stay)

        R = p_sub_optimal * state.gittins + p_opt_back * (state.gittins + opt_s.gittins) + opt_s.gittins * t_opt_expect
        W = p_sub_optimal + p_opt_back * 2 + p_opt_back * t_opt_expect

        state.gittins = R / W
