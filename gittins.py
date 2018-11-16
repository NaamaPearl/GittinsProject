from copy import copy
import numpy as np


class RewardedState:
    def __init__(self, state, r=0):
        self.state = state
        self.gittins = r

    def __gt__(self, other):
        return self.gittins > other.gittins


# calculate new transition probabilities, after optimal state omission (invoked after removal)
def CalcNewProb(opt_s, RS_list, P):
    for state1 in RS_list:
        s1 = state1.state.idx
        if P[s1, opt_s] > 0:  # only update P for states from which opt_s is reachable
            for state2 in RS_list:
                s2 = state2.state.idx
                P[s1, s2] += (P[s1, opt_s] * P[opt_s, s2] / (1 - P[opt_s, opt_s]))

    # zero out transitions to/ from opt_state
    P[opt_s] = 0
    P[:, opt_s] = 0


def CalcIndex(P, state, opt_s):
    state_idx = state.state.idx
    opt_state_idx = opt_s.state.idx

    # in case we haven't visited this state yet
    if np.sum(P[state_idx, :]) == 0:  # TODO: T bored
        return 100

    p_sub_optimal = 1 - P[state_idx, opt_state_idx]
    p_opt_stay = P[opt_state_idx, opt_state_idx] + 10 ** -5
    sum_p_opt = 1 / (1 - p_opt_stay)
    t_opt_expect = 1 / (2 * (1 - p_opt_stay) ** 2)
    p_opt_back = P[state_idx, opt_state_idx] * sum_p_opt * (np.sum(P[opt_state_idx, :]) - p_opt_stay)

    R = p_sub_optimal * state.gittins + p_opt_back * (state.gittins + opt_s.gittins) + opt_s.gittins * t_opt_expect
    W = p_sub_optimal + p_opt_back * 2 + p_opt_back * t_opt_expect

    state.gittins = R / W


def Initiate(model, approximation):
    if approximation:
        P = model.P_hat
        R = model.r_hat
    else:
        P = model.P
        R = model.r

    return copy(P), [RewardedState(state, R[i]) for i, state in enumerate(model.s)]


def Gittins(model, approximation=True):
    result = {}
    score = 1

    P, rs_list = Initiate(model, approximation)
    while len(rs_list) > 1:
        opt_state = max(rs_list)
        rs_list.remove(opt_state)
        result[opt_state.state.idx] = score  # add opt_state to result
        score += 1

        for rewarded_state in rs_list:
            CalcIndex(P, rewarded_state, opt_state)  # calc index after omission, for all remaining states

        CalcNewProb(opt_state.state.idx, rs_list, P)  # calc new transition matrix

    result[rs_list.pop().state.idx] = score  # when only one state remains, simply add it to the result list
    # print(result)
    return result
