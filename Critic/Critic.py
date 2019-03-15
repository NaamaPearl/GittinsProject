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
        while good_agents < kwargs['good_agents']:
            agent = Agent(0, kwargs['initial_state'])
            agent.curr_state = self.model.states[self.model.GetNextState(agent.curr_state.policy_action)]
            if agent.curr_state.chain not in kwargs['active_chains']:
                continue

            good_agents += 1
            for i in range(1, kwargs['trajectory_len']+1):
                new_reward = self.model.GetReward(agent.curr_state.policy_action)
                reward += (new_reward * kwargs['gamma'] ** i)
                agent.curr_state = self.model.states[self.model.GetNextState(agent.curr_state.policy_action)]
        return reward * (kwargs['active_chains_ratio'] / kwargs['good_agents'])  # assume next states after initial are evenly
        # distributed between chains


class OnlinePolicyEvaluator(Evaluator):
    def EvaluatePolicy(self, **kwargs):
        def sum_reward_vec(reward_vec):
            return reduce(lambda a, b: a + b, reward_vec)

        accumulated_diff = sum_reward_vec(kwargs['optimal_agents_reward']) - sum_reward_vec(kwargs['agents_reward'])
        return accumulated_diff / kwargs['running_agents']


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
        self.bad_activated_states = []

    def Update(self, chain, state_idx):
        pass

    def CriticEvaluate(self, **kwargs):
        if kwargs.get('bad_activated_states') is not None:
            self.bad_activated_states.append(kwargs['bad_activated_states'])
        for eval_type in self.eval_type_list:
            evaluated_reward = self.evaluator_dict[eval_type].EvaluatePolicy(**kwargs)
            self.value_vec[eval_type].append(evaluated_reward)


class ChainMDPCritic(Critic):
    def __init__(self, chain_num, **kwargs):
        self.chain_num = chain_num
        self.chain_activations = self.buildChainVec()
        super().__init__(**kwargs)

    def buildChainVec(self):
        return [0 for _ in range(self.chain_num)]

    def Update(self, chain, state_idx):
        if chain is not None:
            self.chain_activations[chain] += 1
