import numpy as np
from functools import reduce


class StateActionPair:
    P_hat_mat = []
    Q_hat_mat = []
    TD_error_mat = []
    r_hat_mat = []
    T_bored_num = 5
    visitations_mat = []

    def __str__(self):
        return 'state #' + str(self.state.idx) + ', action #' + str(self.action)

    def __init__(self, state, action):
        self.state: SimulatedState = state
        self.action = action

    def __gt__(self, other):
        assert isinstance(other, StateActionPair)
        return self.Q_hat > other.Q_hat

    def __eq__(self, other):
        return self.action == other.action and self.state.idx == other.state.idx

    def __hash__(self):
        return hash((self.state, self.action, self.visitations))

    @property
    def chain(self):
        return self.state.chain

    @property
    def r_hat(self):
        return StateActionPair.r_hat_mat[self.state.idx][self.action]

    @r_hat.setter
    def r_hat(self, new_val):
        StateActionPair.r_hat_mat[self.state.idx][self.action] = new_val

    @property
    def P_hat(self):
        return StateActionPair.P_hat_mat[self.state.idx][self.action]

    @P_hat.setter
    def P_hat(self, new_val):
        StateActionPair.P_hat_mat[self.state.idx][self.action] = new_val

    @property
    def visitations(self):
        return StateActionPair.visitations_mat[self.state.idx][self.action]

    @visitations.setter
    def visitations(self, new_val):
        StateActionPair.visitations_mat[self.state.idx][self.action] = new_val

    def UpdateVisits(self):
        self.visitations += 1

    @property
    def Q_hat(self):
        return StateActionPair.Q_hat_mat[self.state.idx][self.action]

    @Q_hat.setter
    def Q_hat(self, new_val):
        StateActionPair.Q_hat_mat[self.state.idx][self.action] = new_val

    @property
    def TD_error(self):
        return StateActionPair.TD_error_mat[self.state.idx][self.action]

    @TD_error.setter
    def TD_error(self, new_val):
        StateActionPair.TD_error_mat[self.state.idx][self.action] = new_val


class StateActionScore:
    score_mat = []

    def __init__(self, state_action):
        self.state_idx = state_action.state.idx
        self.action = state_action.action

    @property
    def score(self):
        return StateActionScore.score_mat[self.state_idx][self.action]

    @score.setter
    def score(self, new_val):
        StateActionScore.score_mat[self.state_idx][self.action] = new_val

    @property
    def visitations(self):
        return StateActionPair.visitations_mat[self.state_idx][self.action]

    def __gt__(self, other):
        if self.score > other.score:
            return True
        if self.score < other.score:
            return False
        return self.visitations < other.visitations

    def __lt__(self, other):
        if self.score < other.score:
            return True
        if self.score > other.score:
            return False
        return self.visitations > other.visitations

    def __str__(self):
        return 'score is: ' + str(self.score) + ', visited ' + str(self.visitations) + ' times'


class SimulatedState:
    """Represents a state with a list of possible actions from current state"""
    V_hat_vec = []
    action_num = 0
    policy = []

    def __init__(self, idx, chain):
        self.idx: int = idx
        self.actions = [StateActionPair(self, a) for a in range(SimulatedState.action_num)]
        self.chain = chain
        self.predecessor = set()

    def __str__(self):
        return 'state #' + str(self.idx) + ' #visitations ' + str(self.visitations)

    @property
    def V_hat(self):
        return SimulatedState.V_hat_vec[self.idx]

    @V_hat.setter
    def V_hat(self, new_val):
        SimulatedState.V_hat_vec[self.idx] = new_val

    @property
    def policy_action(self):
        return self.actions[SimulatedState.policy[self.idx]]

    @policy_action.setter
    def policy_action(self, new_val):
        SimulatedState.policy[self.idx] = new_val

    @property
    def s_a_visits(self):
        return self.policy_action.visitations

    @property
    def visitations(self):
        """ sums visitation counter for all related state-action pairs"""
        return reduce(lambda a, b: a + b, [state_action.visitations for state_action in self.actions])

    @property
    def min_visitations(self):
        visitations_list = [state_action.visitations for state_action in self.actions]
        min_idx = int(np.argmin(visitations_list))
        return visitations_list[min_idx], min_idx

    @property
    def best_action(self):
        try:
            visited_actions = [action for action in self.actions if action.visitations > 0]
            return max(visited_actions)
        except ValueError:
            return np.random.choice(self.actions)

    @property
    def highest_error_action(self):
        return max(self.actions, key=lambda x: abs(x.TD_error))

    @property
    def r_hat(self):
        return self.policy_action.r_hat


class StateScore:
    def __init__(self, state, score):
        self.state = state
        self.score = score

    @property
    def idx(self):
        return self.state.idx

    def __gt__(self, other):
        if self.score > other.score:
            return True
        if self.score < other.score:
            return False
        return self.state.visitations < other.state.visitations


class EvaluatedModel:
    def __init__(self):
        self.r_hat = None
        self.P_hat = None
        self.V_hat = None
        self.Q_hat = None
        self.TD_error = None
        self.visitations = None

    def ResetData(self, state_num, actions):
        self.r_hat = np.zeros((state_num, actions))
        self.P_hat = [np.zeros((actions, state_num)) for _ in range(state_num)]
        self.V_hat = np.zeros(state_num)
        self.Q_hat = np.zeros((state_num, actions))
        self.TD_error = np.zeros((state_num, actions))
        self.visitations = np.zeros((state_num, actions))
