from MDPModel.MDPModel import *
from MDPModel.MDPBasics import *


class SimulatedModel:
    def __init__(self, mdp_model):
        self.MDP_model: MDPModel = mdp_model
        SimulatedState.action_num = mdp_model.actions
        self.policy_dynamics = np.zeros((mdp_model.n, mdp_model.n))
        self.policy_expected_rewards = np.zeros(mdp_model.n)
        self.states = [SimulatedState(idx, self.FindChain(idx)) for idx in range(mdp_model.n)]

    def CalcPolicyData(self, policy):
        self.policy_dynamics, self.policy_expected_rewards = self.MDP_model.CalcPolicyData(policy)

    def GetNextState(self, state_action):
        return np.random.choice(range(self.MDP_model.n), p=self.MDP_model.P[state_action.state.idx][state_action.action])

    def calculate_V(self, gamma):
        return np.linalg.inv(np.eye(self.MDP_model.n) - gamma * self.policy_dynamics) @ self.policy_expected_rewards

    def GetReward(self, state_action):
        return self.MDP_model.r[state_action.state.idx][state_action.action].GiveReward()

    @property
    def n(self):
        return self.MDP_model.n

    @property
    def actions(self):
        return self.MDP_model.actions

    @property
    def init_prob(self):
        return self.MDP_model.init_prob

    def FindChain(self, idx):
        return self.MDP_model.FindChain(idx)

    @property
    def chain_num(self):
        return self.MDP_model.chain_num

    @property
    def init_states_idx(self):
        return self.MDP_model.init_states_idx

    @property
    def type(self):
        return self.MDP_model.type


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

    def getOnlineAndZero(self):
        res = self.accumulated_reward
        self.accumulated_reward = 0
        return res

    @property
    def chain(self):
        return self.curr_state.chain
