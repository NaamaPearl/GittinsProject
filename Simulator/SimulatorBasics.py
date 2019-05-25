from MDPModel.MDPModel import MDPModel
from MDPModel.MDPBasics import SimulatedState
from functools import reduce
from itertools import product
from Simulator.Simulator import SimInputFactory, SimulatorFactory
import numpy as np
import random


class Runner:
    """
    Auxiliary class which holds executes all the requested runs, and outputs their result.
    Initiation requires all simulation parameters, which are shared between all runs.
    Executing also requires an iterable of all MDPs.
    """
    sim_params = None
    varied_definition_str = None
    gt_compare = None

    def __init__(self, sim_params, varied_definition_str, gt_compare=False):
        Runner.varied_definition_str = varied_definition_str
        Runner.sim_params = sim_params
        Runner.gt_compare = gt_compare

        def create_definitions(param):
            return list(product([param], sim_params['method_dict'][param], sim_params[varied_definition_str]))

        self.definitions = reduce(lambda a, b: a + b, map(create_definitions, sim_params['method_dict'].keys()))

    def run(self, mdp_list):
        """Executes a run for each mdp stored"""

        def run_mdp(mdp_idx, mdp):
            """Executes required runs for a single MDP. Creates and returns an updated MDPResult objects"""
            print(f'run MDP num {mdp_idx}')
            return MDPResult(mdp).run(self.definitions)

        return list(map(lambda e: run_mdp(*e), enumerate(mdp_list)))


class MDPResult:
    """Executes required runs for a single MDP, and holds results, alongside it's optimal reward"""

    def __init__(self, mdp):
        self.mdp: MDPModel = mdp
        self.optimal_reward = self.mdp.calc_opt_expected_reward()
        self.result = None

    def run(self, definitions):
        def run_mdp_with_defs(method, parameter, varied_definition):
            """Run every configuration for the required amount of times. Returns results"""
            print(f'    running {method}, prioritizing using {parameter}, with {Runner.varied_definition_str} = '
                  f'{varied_definition}:')
            Runner.sim_params[Runner.varied_definition_str] = varied_definition
            res = ResFactory.generate(Runner.gt_compare, self.mdp)

            for idx in range(1, Runner.sim_params['runs_per_mdp'] + 1):
                print(f'     start run #{idx}')
                simulator = SimulatorFactory(self.mdp, Runner.sim_params, Runner.gt_compare)
                res.update(simulator.simulate((SimInputFactory(method, parameter, Runner.sim_params))))

            res.summarize_critics()
            return res

        self.result = {definition: run_mdp_with_defs(*definition) for definition in definitions}
        return self


class RunResult:
    """Container for the results of all runs of a single configuration"""

    def __init__(self, mdp):
        self.mdp_type = mdp.type
        self.critics = []

    def update(self, critics):
        """Add new result to the container"""
        self.critics.append(critics)

    def summarize_critics(self):
        """After all runs of the configuration were made, summarize results, in preparations for presentation"""
        def critic_getter_generator(attr_name):
            return list(map(lambda x: x.value_vec[attr_name], self.critics))
        res = {
            'online': (np.cumsum(np.mean(critic_getter_generator('online')), axis=0),
                       np.std(critic_getter_generator('online'), axis=0)),
            'offline': (np.mean(critic_getter_generator('offline'), axis=0),
                        np.std(critic_getter_generator('offline'), axis=0)),
            'bad_states': list(map(lambda critic: np.diff(critic.bad_activated_states), self.critics)),
            'critics': self.critics}

        if self.mdp_type in ['chains', 'bridge']:
            res['chain_activations'] = np.mean(list(map(lambda critic: critic.chain_activations, self.critics)), axis=0)

        return res


class GTRunResult(RunResult):
    """In Ground Truth runs we also track calculated Gittins indices versus true indices"""

    def __init__(self, mdp):
        super().__init__(mdp)
        self.indices = []
        self.gt = []

    def update(self, critics, **kwargs):
        """Add new result to the container"""
        super().update(critics)
        self.indices.append(kwargs['indices'])
        self.gt.append(kwargs['gt'])


class ResFactory:
    @staticmethod
    def generate(gt_compare, mdp):
        return GTRunResult(mdp) if gt_compare else RunResult(mdp)


class SimulatedModel:
    def __init__(self, mdp_model):
        self.MDP: MDPModel = mdp_model
        SimulatedState.action_num = mdp_model.actions
        self.policy_dynamics = np.zeros((mdp_model.n, mdp_model.n))
        self.policy_expected_rewards = np.zeros(mdp_model.n)
        self.states = [SimulatedState(idx, self.find_chain(idx)) for idx in range(mdp_model.n)]

    def calc_policy_data(self, policy):
        self.policy_dynamics, self.policy_expected_rewards = self.MDP.calc_policy_data(policy)

    def calculate_v(self, gamma):
        return np.linalg.inv(np.eye(self.MDP.n) - gamma * self.policy_dynamics) @ self.policy_expected_rewards

    def get_reward(self, state_action):
        return self.MDP.generate_reward(state_action.state.idx, state_action.action)

    def get_next_state(self, state_action):
        return self.MDP.get_next_state(state_action.state.idx, state_action.action)

    @property
    def n(self):
        return self.MDP.n

    @property
    def actions(self):
        return self.MDP.actions

    @property
    def init_prob(self):
        return self.MDP.init_prob

    def find_chain(self, idx):
        return self.MDP.find_chain(idx)

    @property
    def chain_num(self):
        return self.MDP.chain_num

    @property
    def init_states_idx(self):
        try:
            return self.MDP.init_states_idx
        except AttributeError:
            raise IOError(f"Initial states are not available in {self.type} MDPs")

    @property
    def type(self):
        return self.MDP.type


class Agent:
    def __init__(self, idx, init_state, agent_type='regular'):
        self.type = agent_type
        self.idx = idx
        self.curr_state = init_state
        self.accumulated_reward = 0
        self.last_activation = 0

    def update(self, new_reward, next_state: SimulatedState):
        self.accumulated_reward += new_reward
        self.curr_state = next_state

    def __lt__(self, other):
        return random.choice([True, False])

    def get_online_and_zero(self):
        res = self.accumulated_reward
        self.accumulated_reward = 0
        return res

    @property
    def chain(self):
        return self.curr_state.chain
