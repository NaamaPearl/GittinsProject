import numpy as np


class RewardGenerator:
    def __init__(self, reward=0):
        self.expected_reward = reward

    def give_reward(self):
        return self.expected_reward

    def __str__(self):
        return f'expected reward = {self.expected_reward}'


class RandomRewardGenerator(RewardGenerator):
    def __init__(self, **kwargs):
        super().__init__()
        self.bernoulli_p = kwargs.get('bernoulli_p', 1)
        self.gauss_mu = np.random.normal(kwargs['gauss_params'][0][0], kwargs['gauss_params'][0][1])
        self.gauss_sigma = kwargs['gauss_params'][1]

        self.expected_reward = self.gauss_mu * self.bernoulli_p

    def give_reward(self):
        return np.random.binomial(1, self.bernoulli_p) * np.random.normal(self.gauss_mu, self.gauss_sigma)

    def __str__(self):
        return ' '.join((super().__str__(),
                        f", bernoulli={self.bernoulli_p}, gauss_exp={self.gauss_mu}, gauss_var={self.gauss_sigma}"))


class RewardGeneratorFactory:
    @staticmethod
    def generate(**kwargs):
        return RandomRewardGenerator(gauss_params=kwargs['reward_params']['gauss_params'],
                                     bernoulli_p=kwargs['reward_params'].get('bernoulli_p'))
