from collections import namedtuple
import numpy as np
from Framework.config import SimulationParameters


class ProblemInput:
    def __init__(self, sim_params: SimulationParameters, mdp_model):
        self.MDP_model = mdp_model
        self.agent_num = sim_params.agents[1]
        self.gamma = mdp_model.MDP.gamma
        self.epsilon = sim_params.epsilon
        self.eval_type = sim_params.eval_type


class SimulationInput:
    def __init__(self, sim_params):
        self.steps = sim_params.steps
        self.reset_freq = sim_params.reset_freq
        self.grades_freq = sim_params.grades_freq
        self.evaluate_freq = sim_params.eval_freq
        self.agents_to_run = sim_params.agents[0]
        self.trajectory_len = sim_params.trajectory_len
        self.T_board = sim_params.T_board


class AgentSimulationInput(SimulationInput):
    def __init__(self, method, prioritizer, parameter, sim_params):
        super().__init__(sim_params)
        self.prioritizer = prioritizer
        self.method = method
        self.parameter = parameter
        self.temporal_extension = sim_params.temporal_extension
        self.gittins_discount = sim_params.gittins_discount
        self.trajectory_num = sim_params.trajectory_num
        self.max_trajectory_len = sim_params.max_trajectory_len


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
