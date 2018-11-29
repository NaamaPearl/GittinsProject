import numpy as np
import random


class SimulatorInput:
    def __init__(self, MDP_model, param='reward', agent_num=5, gamma=0.9, epsilon=0.01):
        self.MDP_model = MDP_model
        self.agent_num = agent_num
        self.gamma = gamma
        self.epsilon = epsilon
        self.parameter = param


class PrioritizedObject:
    """
    Represents an object with a prioritization
    """

    def __init__(self, obj, r=None):
        self.object = obj
        if r is not None:
            self.reward = r
        else:
            self.reward = random.random()

    def __gt__(self, other):
        return self.reward > other.reward

    def __lt__(self, other):
        return self.reward < other.reward

    def __eq__(self, other):
        return self.reward == other.reward

    def __hash__(self):
        return hash(object)

    @property
    def idx(self):
        return self.object.idx
