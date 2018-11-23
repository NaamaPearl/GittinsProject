import numpy as np


class SimulatorInput:
    def __init__(self, MDP_model, agent_num=5, init_prob=None, gamma=0.9):
        self.MDP_model = MDP_model
        self.agent_num = agent_num
        self.gamma = gamma
        if init_prob is not None:
            assert np.sum(init_prob) == 1
            self.init_prob = init_prob
        else:
            self.init_prob = np.ones(MDP_model.n) / MDP_model.n


class PrioritizedObject:
    """
    Represents an object with a prioritization
    """

    def __init__(self, obj, r=0):
        self.object = obj
        self.reward = r

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
