import numpy as np
import random
from collections import namedtuple


class ProblemInput:
    def __init__(self, MDP_model, agent_num, eval_type, gamma, epsilon=0.1):
        self.MDP_model = MDP_model
        self.agent_num = agent_num
        self.gamma = gamma
        self.epsilon = epsilon
        self.eval_type = eval_type


class SimulationInput:
    def __init__(self, steps, agents_to_run, trajectory_len, eval_freq=50, reset_freq=50, grades_freq=10):
        self.steps = steps
        self.reset_freq = reset_freq
        self.grades_freq = grades_freq
        self.evaluate_freq = eval_freq
        self.agents_to_run = agents_to_run
        self.trajectory_len = trajectory_len


class ChainSimulationOutput:
    def __init__(self):
        self.chain_activation = 0
        self.reward_eval = 0


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
    def __init__(self, prioritizer, steps, parameter, agents_to_run, trajectory_len, eval_freq=50):
        super().__init__(steps, agents_to_run, eval_freq, trajectory_len)
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
