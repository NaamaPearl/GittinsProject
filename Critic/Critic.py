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
        # tunnel_reward = {True: 0, False: 0}
        while good_agents < kwargs['good_agents']:
            agent = Agent(0, kwargs['initial_state'])
            agent.curr_state = self.model.states[self.model.GetNextState(agent.curr_state.policy_action)]
            if agent.curr_state.chain not in kwargs['active_chains']:
                continue

            good_agents += 1
            for i in range(1, kwargs['trajectory_len']+1):
                new_reward = self.model.GetReward(agent.curr_state.policy_action)
                reward += (new_reward * kwargs['gamma'] ** (i + 1))
                # tunnel_reward[agent.curr_state.idx in [42, 43, 44, 45, 46]] += new_reward
                agent.curr_state = self.model.states[self.model.GetNextState(agent.curr_state.policy_action)]
        return reward / (kwargs['good_agents'] * kwargs['chain_num']), 0  # assume next states after initial are evenly
        # distributed between chains


class OnlinePolicyEvaluator(Evaluator):
    @staticmethod
    def EvaluatePolicy(**kwargs):
        return reduce(lambda a, b: a + b, kwargs['agents_reward']) / kwargs['running_agents'], 0


class EvaluatorFactory:
    @staticmethod
    def Generate(eval_type, **kwargs):
        if eval_type == 'online':
            return OnlinePolicyEvaluator()
        if eval_type == 'offline':
            return OfflinePolicyEvaluator(kwargs['model'])

    def GenEvaluatorDict(self, eval_type_list, **kwargs):
        return {eval_type: self.Generate(eval_type, **kwargs) for eval_type in eval_type_list}


class CriticFactory:
    @staticmethod
    def Generate(**kwargs):
        model = kwargs['model']
        if model.type == 'chains':
            return ChainMDPCritic(model.chain_num, **kwargs)

        return Critic(**kwargs)  # default


class Critic:
    def __init__(self, **kwargs):
        self.eval_type_list = kwargs['evaluator_type']
        self.evaluator_dict = EvaluatorFactory().GenEvaluatorDict(kwargs['evaluator_type'], **kwargs)
        self.value_vec = {eval_type: [] for eval_type in self.eval_type_list}
        self.reward_tunnel = {eval_type: [] for eval_type in self.eval_type_list}
        self.Reset()

    def Update(self, chain):
        pass

    def CriticEvaluate(self, **kwargs):
        for eval_type in self.eval_type_list:
            evaluated_reward, reward_tunnel = self.evaluator_dict[eval_type].EvaluatePolicy(**kwargs)
            self.value_vec[eval_type].append(evaluated_reward)
            # self.reward_tunnel[eval_type].append(reward_tunnel)

    def Reset(self):
        self.value_vec = {eval_type: [] for eval_type in self.eval_type_list}


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

    def CriticEvalbuate(self, **kwargs):
        super().CriticEvaluate(**kwargs)
        [self.time_chain_activation[chain].append(self.chain_activations[chain]) for chain in range(self.chain_num)]
