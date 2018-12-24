import numpy as np
from functools import reduce
from abc import abstractmethod
from Simulator.SimulatorBasics import Agent


class PolicyEvaluator:
    @staticmethod
    def EvaluatePolicy(**kwargs):
        return {'online': reduce(lambda a, b: a + b, kwargs['agents_reward']) / kwargs['running_agents']}


class OfflinePolicyEvaluator(PolicyEvaluator):
    def __init__(self, model):
        self.model = model

    def EvaluatePolicy(self, **kwargs):
        res_dict = super().EvaluatePolicy(**kwargs)
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
        res_dict['offline'] = reward / good_agents
        return res_dict


class EvaluatorFactory:
    @staticmethod
    def Generate(evaluator_type, **kwargs):
        if 'offline' in evaluator_type:
            return OfflinePolicyEvaluator(kwargs['model'])
        else:
            return PolicyEvaluator()


class CriticFactory:
    @staticmethod
    def Generate(**kwargs):
        model = kwargs['model']
        if model.type == 'chains':
            return ChainMDPCritic(model.chain_num, **kwargs)

        return Critic(**kwargs)  # default


class Critic:
    def __init__(self, **kwargs):
        self.evaluator: PolicyEvaluator = EvaluatorFactory().Generate(**kwargs)
        self.value_vec = {eval_type: [] for eval_type in kwargs['evaluator_type']}

        self.Reset(**kwargs)

    def Update(self, chain):
        pass

    def Evaluate(self, **kwargs):
        eval_res = self.evaluator.EvaluatePolicy(**kwargs)
        [self.value_vec[key].append(eval_res[key]) for key in eval_res.keys()]

    def Reset(self, **kwargs):
        self.value_vec = {eval_type: [] for eval_type in kwargs['evaluator_type']}


class ChainMDPCritic(Critic):
    def __init__(self, chain_num, **kwargs):
        self.chain_num = chain_num
        self.chain_activations = None
        self.time_chain_activation = None
        super().__init__(**kwargs)

    def Update(self, chain):
        if chain is not None:
            self.chain_activations[chain] += 1

    def Reset(self, **kwargs):
        super().Reset(**kwargs)
        self.chain_activations = [0 for _ in range(self.chain_num)]
        self.time_chain_activation = [[] for _ in range(self.chain_num)]

    def Evaluate(self, **kwargs):
        super().Evaluate(**kwargs)
        [self.time_chain_activation[chain].append(self.chain_activations[chain]) for chain in range(self.chain_num)]
