import numpy as np


class RewardGenerator:
    def __init__(self, reward=0):
        self.expected_reward = reward

    def GiveReward(self):
        return self.expected_reward


class RandomRewardGenerator(RewardGenerator):
    def __init__(self, **kwargs):
        super().__init__()
        if kwargs.get('bernoulli_p') is None:
            self.bernoulli_p = 1
        else:
            self.bernoulli_p = kwargs['bernoulli_p']

        self.gauss_mu = np.random.normal(kwargs['gauss_params'][0][0], kwargs['gauss_params'][0][1])
        self.gauss_sigma = kwargs['gauss_params'][1]

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
