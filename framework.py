import numpy as np
import random
from collections import namedtuple


class ProblemInput:
    def __init__(self, **kwargs):
        self.MDP_model = kwargs['MDP_model']
        self.agent_num = kwargs['agent_num']
        self.gamma = kwargs['gamma']
        self.epsilon = kwargs['epsilon']
        self.eval_type = kwargs['eval_type']


class SimulationInput:
    def __init__(self, **kwargs):
        self.steps = kwargs['steps']
        self.reset_freq = kwargs['reset_freq']
        self.grades_freq = kwargs['grades_freq']
        self.evaluate_freq = kwargs['eval_freq']
        self.agents_to_run = kwargs['agents_to_run']
        self.trajectory_len = kwargs['trajectory_len']


class ChainSimulationOutput:
    def __init__(self):
        self.chain_activation = 0
        self.reward_eval = []


SimulationData = namedtuple('SimulationData', ('input', 'output'))


class Agent:
    def __init__(self, idx, init_state):
        self.idx = idx
        self.curr_state = init_state
        self.accumulated_reward = 0

    def __lt__(self, other):
        return random.choice([True, False])

    @property
    def chain(self):
        return self.curr_state.chain


class AgentSimulationInput(SimulationInput):
    def __init__(self, prioritizer, parameter, **kwargs):
        super().__init__(**kwargs)
        self.prioritizer = prioritizer
        self.parameter = parameter


class EvaluatedModel:
    def __init__(self):
        self.r_hat = None
        self.P_hat = None
        self.V_hat = None
        self.Q_hat = None
        self.TD_error = None
        self.visitations = None

    def ResetData(self, state_num, actions):
        self.r_hat = np.zeros((state_num, actions))
        self.P_hat = [np.zeros((actions, state_num)) for _ in range(state_num)]
        self.V_hat = np.zeros(state_num)
        self.Q_hat = np.zeros((state_num, actions))
        self.TD_error = np.zeros((state_num, actions))
        self.visitations = np.zeros((state_num, actions))


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

    def __str__(self):
        return str(self.object) + str(self.reward)

    @property
    def idx(self):
        return self.object.idx


class SweepingPrioObject(PrioritizedObject):
    def __init__(self, obj, r):
        super().__init__(obj, r)
        # self.active = True

    def __gt__(self, other):
        return super().__gt__(other)
        # return self.active * super().__gt__(other)
