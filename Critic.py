class CriticFactory:
    @staticmethod
    def Generate(**kwargs):
        if kwargs['type'] == 'chains':
            return ChainMDPCritic(kwargs['chain_num'])

        return Critic()  # default


class Critic:
    def Update(self, chain):
        pass


class ChainMDPCritic(Critic):
    def __init__(self, chain_num):
        self.chain_activations = [0 for _ in range(chain_num)]

    def Update(self, chain):
        self.chain_activations[chain] += 1
