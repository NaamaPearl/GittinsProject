import heapq
from collections import namedtuple
import numpy as np
import MDPModel

RewardState = namedtuple('reward_state', ('reward', 'state', 'index'))


def Heapify(model):
    # create a heap containing all states, sorted according to their estimated reward
    heap = [RewardState(model.r_hat[i], state, 0) for i, state in enumerate(model.s)]
    heapq._heapify_max(heap)


def CalcProb(state1, state2, opt_state, P):
    return 0  # TODO how to pair state's real index to index to new mat?


def CalcNewProb(opt_state, heap, heap_size, curr_P):
    new_P = np.zeros((heap_size, heap_size))
    states = set(heapq.nlargest(heap_size, heap))
    for state1 in states:
        for state2 in states:
            new_P[state1, state2] = CalcProb(state1, state2, opt_state, curr_P)

    return new_P
