from dataclasses import dataclass, field
from typing import List, Set, Dict

"""
Parameters are ordered by hierarchy.  
Note that some parameters' default value are derived from the below lists.
You can always add values to the constructor in order to override default values.  
"""

# Default Values for MDP.
n = 46
chain_num = 3
actions = 3
succ_num = 3
op_succ_num = 5
gamma = 0.95

# Tunnel defaults
tunnel_length = 5


@dataclass
class MDPConfig:
    """Needed parameters for MDP generation. Note that some values are hardcoded, so pay attention!"""
    n: int = n
    chain_num: int = chain_num
    actions: int = actions
    succ_num: int = succ_num  # number of successors per state
    op_succ_num: int = op_succ_num  # number of states from which successors are raffled
    gamma: float = gamma
    reward_dict: Dict = field(default_factory=lambda: {})


@dataclass
class TreeMDPConfig(MDPConfig):
    init_state_idx: Set[int] = frozenset({0})
    traps_num: int = 0
    resets_num: int = 0


@dataclass
class CliqueMDPConfig(TreeMDPConfig):
    active_chains: Set[int] = field(default_factory=lambda: {chain_num - 1})  # chains with rewarded state-action pairs
    reward_dict: Dict = field(default_factory=lambda: {chain_num - 1: {'gauss_params': ((10, 4), 0)}})


@dataclass
class StarMDPConfig(CliqueMDPConfig):
    reward_dict: Dict = field(default_factory=lambda: {1: {'gauss_params': ((100, 3), 0)},
                                                       2: {'gauss_params': ((0, 0), 0)},
                                                       3: {'gauss_params': ((100, 2), 0)},
                                                       4: {'gauss_params': ((1, 0), 0)},
                                                       0: {'gauss_params': ((110, 4), 0)}})
    active_chains: Set[int] = field(default_factory=lambda: set(range(chain_num)))


@dataclass
class TunnelMDPConfig(CliqueMDPConfig):
    tunnel_start: int = n
    tunnel_length: int = tunnel_length
    tunnel_indices: List[int] = None
    reward_dict: Dict = field(default_factory=lambda: {chain_num - 1: {'gauss_params': ((10, 4), 0)},
                                                       'lead_to_tunnel': {'gauss_params': ((-1, 0), 0)},
                                                       'tunnel_end': {'gauss_params': ((100, 0), 0)}})

    def __post_init__(self):
        self.tunnel_indices = list(range(self.tunnel_start - self.tunnel_length, self.tunnel_start))


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
