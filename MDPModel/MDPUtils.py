from dataclasses import dataclass, field
from typing import List, Set, Dict


@dataclass
class MDPConfig:
    """Needed parameters for MDP generation. Note that some values are hardcoded, so pay attention!"""
    n: int = 46
    chain_num: int = 3
    actions: int = 3
    succ_num: int = 3  # number of successors per state
    op_succ_num: int = 5  # number of states from which successors are raffled
    gamma: float = 0.95
    reward_dict: Dict = field(default_factory=lambda: {})


@dataclass
class TreeMDPConfig(MDPConfig):
    init_state_idx: Set[int] = frozenset({0})
    traps_num: int = 0
    resets_num: int = 0


@dataclass
class CliqueMDPConfig(TreeMDPConfig):
    active_chains: Set[int] = field(default_factory=lambda: {2})  # chains with rewarded state-action pairs
    reward_dict: Dict = field(default_factory=lambda: {2: {'gauss_params': ((10, 4), 0)}})


@dataclass
class StarMDPConfig(CliqueMDPConfig):
    reward_dict: Dict = field(default_factory=lambda: {1: {'gauss_params': ((100, 3), 0)},
                                                       2: {'gauss_params': ((0, 0), 0)},
                                                       3: {'gauss_params': ((100, 2), 0)},
                                                       4: {'gauss_params': ((1, 0), 0)},
                                                       0: {'gauss_params': ((110, 4), 0)}})
    active_chains: Set[int] = field(default_factory=lambda: set(range(3)))


@dataclass
class TunnelMDPConfig(CliqueMDPConfig):
    tunnel_length: int = 5
    tunnel_indices: List[int] = field(default_factory=lambda: list(range(46 - 5, 46)))
    reward_dict: Dict = field(default_factory=lambda: {2: {'gauss_params': ((10, 4), 0)},
                                                       'lead_to_tunnel': {'gauss_params': ((-1, 0), 0)},
                                                       'tunnel_end': {'gauss_params': ((100, 0), 0)}})


@dataclass
class CliffMDPConfig(TreeMDPConfig):
    chain_num: int = 1
    actions: int = 4
    size: int = 5
    random_prob: float = 0.85

    def __post_init__(self):
        self.n = self.size ** 2  # n is the square of a single edge


@dataclass
class DirectedTreeMDPConfig(TreeMDPConfig):
    reward_dict: Dict = field(default_factory=lambda: {'basic': {'gauss_params': ((0, 1), 0)},
                                                       17: {'gauss_params': ((50, 0), 0)},
                                                       25: {'gauss_params': ((-50, 0), 0)}})
    depth: int = 5
    chain_num = 1
    succ_num: int = 2  # note that root has only 2 sons, so changing this number will cause a bug

    def __post_init__(self):
        self.n = 2 ** self.depth - 1
