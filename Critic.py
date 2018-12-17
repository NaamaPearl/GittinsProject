import numpy as np
from ModelSimulator import Agent, SimulatedModel
from functools import reduce
from abc import abstractmethod


class Evaluator:
    @abstractmethod
    def EvaluatePolicy(self, *args):
        pass


class OfflinePolicyEvaluator(Evaluator):
    def __init__(self, model):
        self.model: SimulatedModel = model

    def EvaluatePolicy(self, trajectory_len):
        reward = 0
        good_agents = 0
        for _ in range(50):
            agent = Agent(0, self.model.states[np.random.choice(list(self.model.init_states_idx))])
            agent.curr_state = self.model.states[self.model.GetNextState(agent.curr_state.policy_action)]
            if agent.curr_state.chain == 0:
                continue

            good_agents += 1
            for _ in range(trajectory_len):
                reward += self.model.GetReward(agent.curr_state.policy_action)
                agent.curr_state = self.model.states[self.model.GetNextState(agent.curr_state.policy_action)]

        return reward / good_agents


class OnlinePolicyEvaluator(Evaluator):
    @staticmethod
    def EvaluatePolicy(agents_reward):
        return reduce(lambda a, b: a + b, agents_reward) / len(agents_reward)


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
        if model.MDP_model.type == 'chains':
            return ChainMDPCritic(model.MDP_model.chain_num, **kwargs)

        return Critic(**kwargs)  # default


class Critic:
    def __init__(self, **kwargs):
        self.evaluator: Evaluator = EvaluatorFactory().Generate(**kwargs)
        self.value_vec = None

        self.Reset()

    def Update(self, chain):
        pass

    def Evaluate(self, *args):
        self.value_vec.append(self.evaluator.EvaluatePolicy(*args))

    def Reset(self):
        self.value_vec = []


class ChainMDPCritic(Critic):
    def __init__(self, chain_num, **kwargs):
        self.chain_num = chain_num
        self.chain_activations = None
        super().__init__(**kwargs)

    def Update(self, chain):
        if chain is not None:
            self.chain_activations[chain] += 1

    def Reset(self):
        self.chain_activations = [0 for _ in range(self.chain_num)]
