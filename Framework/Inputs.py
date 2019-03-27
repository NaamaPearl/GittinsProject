from collections import namedtuple
import numpy as np


class ProblemInput:
    def __init__(self, **kwargs):
        self.MDP_model = kwargs['MDP_model']
        self.agent_num = kwargs['agents'][1]
        self.gamma = kwargs['gamma']
        self.epsilon = kwargs['epsilon']
        self.eval_type = kwargs['eval_type']


class SimulationInput:
    def __init__(self, **kwargs):
        self.steps = kwargs['steps']
        self.reset_freq = kwargs['reset_freq']
        self.grades_freq = kwargs['grades_freq']
        self.evaluate_freq = kwargs['eval_freq']
        self.agents_to_run = kwargs['agents'][0]
        self.trajectory_len = kwargs['trajectory_len']
        self.T_board = kwargs['T_board']


class AgentSimulationInput(SimulationInput):
    def __init__(self, method, prioritizer, parameter, **kwargs):
        super().__init__(**kwargs)
        self.prioritizer = prioritizer
        self.method = method
        self.parameter = parameter
        self.temporal_extension = kwargs['temporal_extension']
        self.gittins_discount = kwargs['gittins_discount']
        self.trajectory_num = kwargs.get('trajectory_num')
        self.max_trajectory_len = kwargs.get('max_trajectory_len')


class RewardEvaluation:
    def __init__(self, eval_type_list):
        self.reward_eval_list = {eval_type: [] for eval_type in eval_type_list}

    def add(self, reward_dict):
        for k, v in reward_dict.items():
            v = np.array(v)
            if k == 'online':
                v = np.cumsum(v)
            self.reward_eval_list[k].append(v)

    def get(self, eval_type):
        return np.array(self.reward_eval_list[eval_type])


SimulationData = namedtuple('SimulationData', ('input', 'output'))
