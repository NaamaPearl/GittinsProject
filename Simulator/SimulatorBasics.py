from MDPModel.MDPModel import MDPModel
from MDPModel.MDPBasics import SimulatedState
import numpy as np
import random


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
