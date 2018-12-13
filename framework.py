import numpy as np
import random

class SimulatorInput:
    def __init__(self, MDP_model, agent_num, gamma=0.9, epsilon=0.1):
        self.MDP_model = MDP_model
        self.agent_num = agent_num
        self.gamma = gamma
        self.epsilon = epsilon


class SimulationInput:
    def __init__(self, prioritizer, steps, parameter, agents_to_run=1):
        self.prioritizer = prioritizer
        self.steps = steps
        self.parameter = parameter
        self.agents_to_run = agents_to_run


class AgentSimulationInput(SimulationInput):
    def __init__(self, prioritizer, steps, parameter, agents_to_run, grades_freq=10, reset_freq=50):
        super().__init__(prioritizer, steps, parameter)
        self.agents_to_run = agents_to_run
        self.grades_freq = grades_freq
        self.reset_freq = reset_freq


class EvaluatedModel:
    def __init__(self):
        self.r_hat = None
        self.P_hat = None
        self.V_hat = None
        self.Q_hat = None
        self.TD_error = None
        self.policy = None

    def ResetData(self, state_num, actions):
        self.r_hat = np.zeros((state_num, actions))
        self.P_hat = [np.zeros((actions, state_num)) for _ in range(state_num)]
        self.policy = [random.randint(0, actions - 1) for _ in range(state_num)]
        self.V_hat = np.zeros(state_num)
        self.Q_hat = np.zeros((state_num,actions))
        self.TD_error = np.zeros((state_num,actions))



class PrioritizedObject:
    """
    Represents an object with a prioritization
    """

    def __init__(self, obj, r):
        self.object = obj
        self.reward = r

    def __gt__(self, other):
        return self.reward > other.reward

    def __lt__(self, other):
        return self.reward < other.reward

    def __hash__(self):
        return hash(object)

    @property
    def idx(self):
        return self.object.idx
