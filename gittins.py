import heapq
from collections import namedtuple
from copy import copy
import numpy as np

RewardState = namedtuple('reward_state', ('gittins', 'state'))


def CalcNewProb(opt_s, heap, P):
    for s1 in heap:
        for s2 in heap:
            P[s1, s2] += (P[s1, opt_s] * P[opt_s, s2] / (1 - P[opt_s, opt_s]))

    # zero out transitions to/ from opt_state
    P[opt_s]    = 0
    P[:, opt_s] = 0
    return P


def CalcIndex(P, state, opt_s):
    p_sub_optimal = np.sum(P[state]) - P[state, opt_s]
    p_opt_stay = P[opt_s, opt_s]
    sum_p_opt = 1 / (1 - p_opt_stay)
    t_opt_expect = 1 / (2 * (1 - p_opt_stay) ** 2)
    p_opt_back = P[state, opt_s] * sum_p_opt * (np.sum[opt_s, :] - p_opt_stay)

    R = p_sub_optimal * state.gittins + p_opt_back * (state.gittins + opt_s.gittins) + opt_s.gittins * t_opt_expect
    W = p_sub_optimal + p_opt_back * 2 + p_opt_back * t_opt_expect

    return R / W


def Gittins(model):
    result = []
    P = copy(model.P_hat)
    rs_list = [RewardState(model.r_hat[i], state) for i, state in enumerate(model.s)]
    while len(rs_list):
        opt_state = max(rs_list)
        rs_list.remove(opt_state)
        result.append(opt_state.state)  # add opt_state to result

        for rs_pair in rs_list:
            rs_pair.gittins = CalcIndex(P, rs_pair, opt_state)

    return result
