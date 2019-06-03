from dataclasses import dataclass, field
from typing import Tuple, List
from pathlib import Path


@dataclass
class SimulationParameters:
    steps: int = 5000
    eval_type = ['online', 'offline']
    agents: List[Tuple[int]] = field(default_factory=lambda: [(10, 30)])
    trajectory_len: int = 150
    eval_freq: int = 50
    epsilon = 0.15
    reset_freq = 20000
    grades_freq = 50
    gittins_discount = 0.9
    temporal_extension: List[int] = field(default_factory=lambda: [1])
    T_board = 3
    runs_per_mdp: int = 3
    varied_param = None
    regular : List[int] = field(default_factory=lambda: [1])
    run_type : str = 'regular'
    trajectory_num = 50
    max_trajectory_len = 50
    method_dict: dict = field(default_factory=lambda: {'random': ['None'], 'greedy': ['reward', 'error'], 'gittins': ['reward', 'error']})
    mdp_types: List[str] = field(default_factory=lambda: [ 'clique', 'tunnel', 'directed', 'cliff'])
    gt_compare: bool = False
    gittins_compare = None
    results_address: str = str(Path.cwd() / 'run_res.pckl')

    def __post_init__(self):
        if self.gt_compare:
            self.gittins_compare = [('model_free', 'error'), ('gittins', 'error')]
            self.method_dict['gittins'].append('ground_truth')
            self.method_dict['gittins'].append('error')
            self.method_dict['model_free'] = ['error']

        if self.run_type != 'temporal_extension':
            if len(self.temporal_extension) != 1:
                raise('varied parameter is not temporal extension but temporal extension is a list')
            self.temporal_extension = self.temporal_extension[0]

        if self.run_type != 'agents':
            if len(self.agents) != 1:
                raise('varied parameter is not agents but agents is a list')
            self.agents = self.agents[0]

        if self.run_type == 'agents':
            self.method_dict = {'gittins': ['error']}

        if self.run_type == 'temporal_extension':
            self.method_dict = {'random': [None], 'gittins': ['error']}