import numpy as np
from functools import reduce
from abc import abstractmethod
from Simulator.SimulatorBasics import Agent


class Evaluator:
    @abstractmethod
    def EvaluatePolicy(self, **kwargs):
        pass


class OfflinePolicyEvaluator(Evaluator):
    def __init__(self, model):
        self.model = model

    def EvaluatePolicy(self, **kwargs):
        reward = 0
        good_agents = 0
        for _ in range(50):
            agent = Agent(0, self.model.states[np.random.choice(list(self.model.init_states_idx))])
            agent.curr_state = self.model.states[self.model.GetNextState(agent.curr_state.policy_action)]
            if agent.curr_state.chain == 0:
                continue

            good_agents += 1
            for _ in range(kwargs['trajectory_len']):
                reward += self.model.GetReward(agent.curr_state.policy_action)
                agent.curr_state = self.model.states[self.model.GetNextState(agent.curr_state.policy_action)]

        return reward / good_agents


class OnlinePolicyEvaluator(Evaluator):
    @staticmethod
    def EvaluatePolicy(**kwargs):
        return reduce(lambda a, b: a + b, kwargs['agents_reward']) / kwargs['running_agents']


class EvaluatorFactory:
    @staticmethod
    def Generate(evaluator_type, **kwargs):
        if evaluator_type == 'online':
            return OnlinePolicyEvaluator()
        if evaluator_type == 'offline':
            return OfflinePolicyEvaluator(kwargs['model'])


class CriticFactory:
    @staticmethod
    def Generate(**kwargs):
        model = kwargs['model']
        if model.type == 'chains':
            return ChainMDPCritic(model.chain_num, **kwargs)

        return Critic(**kwargs)  # default


class Critic:
    def __init__(self, **kwargs):
        self.evaluator: Evaluator = EvaluatorFactory().Generate(**kwargs)
        self.value_vec = None

        self.Reset()

    def Update(self, chain):
        pass

    def Evaluate(self, **kwargs):
        self.value_vec.append(self.evaluator.EvaluatePolicy(**kwargs))

    def Reset(self):
        self.value_vec = []


class ChainMDPCritic(Critic):
    def __init__(self, chain_num, **kwargs):
        self.chain_num = chain_num
        self.chain_activations = None
        self.time_chain_activation = None
        super().__init__(**kwargs)

    def Update(self, chain):
        if chain is not None:
            self.chain_activations[chain] += 1

    def Reset(self):
        super().Reset()
        self.chain_activations = [0 for _ in range(self.chain_num)]
        self.time_chain_activation = [[] for _ in range(self.chain_num)]

    def Evaluate(self, **kwargs):
        super().Evaluate(**kwargs)
        [self.time_chain_activation[chain].append(self.chain_activations[chain]) for chain in range(self.chain_num)]
