from abc import abstractmethod
import numpy as np
import random


class RewardGenerator:
    def __init__(self):
        self.expected_reward = 0

    @abstractmethod
    def GiveReward(self):
        return 0


class RandomRewardGenerator(RewardGenerator):
    def __init__(self, **kwargs):
        super().__init__()
        if kwargs.get('bernoulli_p') is None:
            self.bernoulli_p = 1
        else:
            self.bernoulli_p = kwargs['bernoulli_p']

        try:
            self.gauss_mu = np.random.normal(kwargs['gauss_params'][0][0], kwargs['gauss_params'][0][1])
            self.gauss_sigma = kwargs['gauss_params'][1]

        except KeyError:
            self.gauss_mu = np.random.normal(0, 50)
            self.gauss_sigma = abs(np.random.normal(10, 4))

        self.expected_reward = self.gauss_mu * self.bernoulli_p

    def GiveReward(self):
        return np.random.binomial(1, self.bernoulli_p) * np.random.normal(self.gauss_mu, self.gauss_sigma)


class RewardGeneratorFactory:
    @staticmethod
    def Generate(rewarded_state, **kwargs):
        if rewarded_state:
            try:
                return RandomRewardGenerator(gauss_params=kwargs['reward_params']['gauss_params'],
                                             bernoulli_p=kwargs['reward_params'].get('bernoulli_p'))
            except TypeError:
                return RandomRewardGenerator()
        return RewardGenerator()
