from abc import abstractmethod
import numpy as np


class RewardGenerator:
    def __init__(self):
        self.expected_reward = 0

    @abstractmethod
    def GiveReward(self):
        return 0


class RandomRewardGenerator(RewardGenerator):
    def __init__(self, gauss_params, bernoulli_p):
        super().__init__()
        self.bernoulli_p = bernoulli_p
        self.gauss_mu = np.random.normal(gauss_params[0][0], gauss_params[0][1])
        self.gauss_sigma = gauss_params[1]
        self.expected_reward = self.gauss_mu * self.bernoulli_p

    def GiveReward(self):
        return np.random.binomial(1, self.bernoulli_p) * np.random.normal(self.gauss_mu, self.gauss_sigma)


class RewardGeneratorFactory:
    @staticmethod
    def Generate(rewarded_state, **kwargs):
        if rewarded_state:
            return RandomRewardGenerator(gauss_params=kwargs['reward_params']['gauss_params'],
                                         bernoulli_p=kwargs['reward_params']['bernoulli_p'])
        return RewardGenerator()
