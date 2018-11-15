import numpy as np
import copy


class MDPModel(object):
    def __init__(self, n=10, gamma=0.9, loc=0.5, var=0.2):
        self.n = n
        self.P = self.gen_P_matrix(loc, var)
        self.P_hat = np.zeros((n, n))
        self.r = np.random.random_integers(low=0, high=100, size=n)  # TODO stochastic reward
        self.r_hat = np.zeros(n)
        self.s = [State(i) for i in range(n)]
        self.V = np.zeros(n)
        self.gamma = gamma

    def gen_P_matrix(self, loc, var):
        P = np.array([self.get_row_of_P(self.n, loc, var) for _ in range(self.n)])
        return P

    @staticmethod
    def get_row_of_P(n, loc, var):
        row = np.random.random(n)  # TODO: variance doesn't calculated as well
        row /= row.sum()
        return row

    def simulate_one_step(self, curr_agent):
        curr_s = self.get_state(curr_agent.curr_state)

        if curr_agent.set_init_state:
            curr_s.update_visits()
            curr_agent.reset_init_state()

        next_s = self.choose_next_state(curr_s)
        r = self.r[next_s.idx]

        self.update_reward(next_s, r)
        self.update_p(curr_s, next_s)
        self.update_V(curr_s.idx)
        curr_agent.update_state(next_s.idx)
        next_s.update_visits()

        return r, next_s.idx

    def choose_next_state(self, curr_s):
        next_s_idx = np.random.choice(np.arange(self.n), p=self.P[curr_s.idx])
        return self.get_state(next_s_idx)

    def update_reward(self, next_s, new_reward):
        curr_est_reward = self.r_hat[next_s.idx]
        new_est_reward = (curr_est_reward * next_s.visits + new_reward) / (next_s.visits + 1)

        self.r_hat[next_s.idx] = new_est_reward

    def update_p(self, curr_s, next_s):
        curr_est_p_row = self.P_hat[curr_s.idx]
        curr_num_of_tran = curr_est_p_row * (curr_s.visits - 1)
        curr_num_of_tran[next_s.idx] += 1

        new_est_p_row = curr_num_of_tran / curr_s.visits
        self.P_hat[curr_s.idx] = new_est_p_row

    def get_state(self, n):
        return self.s[n]

    def update_V(self, idx):
        self.V[idx] = self.r_hat[idx] + self.gamma * np.dot(self.V, self.P_hat[idx])

    def P_hat_mean_diff(self):
        return abs(self.P - self.P_hat).mean()


class State(object):
    def __init__(self, idx):
        self.idx = idx
        self.visits = 0

    def update_visits(self):
        self.visits += 1


class Agent(object):
    def __init__(self, idx, init_state):
        self.idx = idx
        self.curr_state = init_state
        self.set_init_state = True

    def update_state(self, new_idx_state):
        self.curr_state = new_idx_state

    def reset_init_state(self):
        self.set_init_state = False


class Simulator(object):
    def __init__(self, MDP_model=None, k=5, init_state=0, choose_agent_fun=None, first_agent=0, **kwargs):
        self.MDP_model = MDP_model if MDP_model is not None else MDPModel(**kwargs)
        self.k = k

        self.curr_agent = first_agent

        if choose_agent_fun is not None:
            self.choose_agent = choose_agent_fun
        else:
            self.choose_agent = lambda x: (x + 1) % k

        self.k = [Agent(i, init_state) for i in range(k)]

    def get_agent(self, k):
        return self.k[k]

    def simulate(self):
        next_agent_idx = self.choose_agent(self.curr_agent)
        next_agent = self.get_agent(next_agent_idx)

        r, next_state = self.MDP_model.simulate_one_step(next_agent)

        self.curr_agent = next_agent.idx

        return r, next_state, next_agent.idx

    def evaluate_P_hat(self):
        return self.MDP_model.P_hat_mean_diff()


if __name__ == '__main__':

    n = 6
    k = 4

    sequence_simulator = Simulator(k=k, n=n)

    MDP = copy.copy(sequence_simulator.MDP_model)


    # random_agent = lambda x: np.random.choice(a=np.arange(k))

    def three_agent(_):
        return 3


    three_simulator = Simulator(k=k, n=n, MDP_model=MDP, choose_agent_fun=three_agent)

    # MDP = MDPModel(n=n)

    for _ in range(100000):
        r_sqe, state_sqe, agent_seq = sequence_simulator.simulate()
        # r_rand, state_rand, agent_rand = three_simulator.simulate()

    print(sequence_simulator.evaluate_P_hat())
    print(three_simulator.evaluate_P_hat())
    print('all done')
