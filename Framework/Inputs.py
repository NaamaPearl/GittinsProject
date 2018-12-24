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


class AgentSimulationInput(SimulationInput):
    def __init__(self, prioritizer, parameter, **kwargs):
        super().__init__(**kwargs)
        self.prioritizer = prioritizer
        self.parameter = parameter
        self.gittins_look_ahead = kwargs['gittins_look_ahead']
        self.gittins_discount = kwargs['gittins_discount']
        self.T_bored = kwargs['T_bored']


class ChainSimulationOutput:
    def __init__(self, eval_type_list):
        self.chain_activation = 0
        self.reward_eval = {eval_type: [] for eval_type in eval_type_list}


SimulationData = namedtuple('SimulationData', ('input', 'output'))