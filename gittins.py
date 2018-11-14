import heapq
from collections import namedtuple
from copy import copy
import numpy as np

RewardState = namedtuple('reward_state', ('gittins', 'state'))


def Heapify(model):
    # create a heap containing all states, sorted according to their estimated reward
    heap = [RewardState(model.r_hat[i], state) for i, state in enumerate(model.s)]
    return heapq._heapify_max(heap)


def CalcNewProb(opt_s, heap, P):
    for state1 in heap:
        s1 = state1.idx
        for state2 in heap:
            s2 = state2.idx
            P[s1, s2] += (P[s1, opt_s] * P[opt_s, s2] / (1 - P[opt_s, opt_s]))

    # zero out transitions to/ from opt_state
    P[opt_s]    = 0
    P[:, opt_s] = 0
    return P


def CalcIndex(P, state, opt_s, indexes):
    p_in = np.sum(P[state]) - P[state, opt_s]
    p_opt = P[opt_s, opt_s]
    p_opt_inf = 1 / (1 - p_opt)
    r_opt_inf = 1 / (2 * (1 - p_opt) ** 2)
    p_out = P[state, opt_s] * p_opt_inf * (np.sum[:, opt_s] - p_opt)

    r_state = indexes[state]
    r_opt = indexes[opt_s]

    index = p_in * r_state + (r_state + r_opt) * p_opt_inf + r_opt * r_opt_inf



def Gittins(model):
    result = []
    states_heap = Heapify(model)
    P = copy(model.P_hat)
    T = np.ones(model.n)
    while len(states_heap):
        opt_state = heapq._heappop_max(states_heap)
        result.append(opt_state)  # add opt_state to result


