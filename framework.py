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
