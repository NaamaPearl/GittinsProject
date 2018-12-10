

class SimulatorInput:
    def __init__(self, MDP_model, agent_num, gamma=0.9, epsilon=0.1):
        self.MDP_model = MDP_model
        self.agent_num = agent_num
        self.gamma = gamma
        self.epsilon = epsilon


class PrioritizedObject:
    """
    Represents an object with a prioritization
    """

    def __init__(self, obj, r):
        self.object = obj
        self.reward = r

    def __gt__(self, other):
        return self.reward > other.reward

    def __lt__(self, other):
        return self.reward < other.reward

    def __hash__(self):
        return hash(object)

    @property
    def idx(self):
        return self.object.idx
