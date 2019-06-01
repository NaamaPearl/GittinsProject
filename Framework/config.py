from dataclasses import dataclass, field
from typing import Tuple, List
from pathlib import Path


@dataclass
class SimulationParameters:
    steps: int = 100
    eval_type = ['online', 'offline']
    agents: Tuple[int] = (10, 30)
    trajectory_len: int = 150
    eval_freq: int = 50
    epsilon = 0.15
    gamma = 0.9
    reset_freq = 20000
    grades_freq = 50
    gittins_discount = 0.9
    temporal_extension = [1]
    T_board = 3
    runs_per_mdp = 1
    varied_param = None
    varied_definition_str: str = 'temporal_extension'
    trajectory_num = 50
    max_trajectory_len = 50
    method_dict = {'greedy': ['error', 'reward']}
    mdp_types: List[str] = field(default_factory=lambda: ['tunnel', 'star', 'clique', 'cliff', 'directed'])
    gt_compare: bool = False
    gittins_compare = None
    results_address: str = str(Path.cwd() / 'run_res.pckl')

    def __post_init__(self):
        if self.gt_compare:
            self.gittins_compare = [('model_free', 'error'), ('gittins', 'error')]
            self.method_dict['gittins'].append('ground_truth')
