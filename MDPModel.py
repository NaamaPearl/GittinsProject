import numpy as np


class MDPModel(object):
    def __init__(self, n=10, k=5, init_state=0, loc=0.5, var=0.2):
        self.n     = n
        self.k     = k
        self.P     = self.gen_P_matrix(loc, var)
        self.P_hat = np.zeros((n, n))
        self.r     = np.random.random_integers(low=0, high=100, size=n)  # TODO stochastic reward
        self.r_hat = np.zeros(n)
        self.s     = [State(i) for i in range(n)]
        self.k     = [Agent(i, init_state) for i in range(k)]

    def gen_P_matrix(self, loc, var):
        P = np.array([self.get_row_of_P(self.n, loc, var) for _ in range(self.n)])
        return P


    @staticmethod
    def get_row_of_P(n, loc, var):
        row = np.random.random(n) #TODO: variance doesn't calculated as well
        row /= row.sum()
        return row

    def simulate_one_step(self, k):
        curr_agent = self.k[k]
        curr_s     = self.s[curr_agent.curr_state]

        if curr_agent.set_init_state:
            curr_s.update_visits()
            curr_agent.reset_init_state()

        next_s     = self.choose_next_state(curr_s)
        r          = self.r[next_s.idx]

        self.update_reward(next_s, r)
        self.update_p(curr_s, next_s)
        curr_agent.update_state(next_s.idx)
        next_s.update_visits()


        return r, next_s.idx

    def choose_next_state(self, curr_s):
        next_s_idx = np.random.choice(np.arange(self.n), p=self.P[curr_s.idx])
        return self.s[next_s_idx]

    def update_reward(self, next_s, new_reward):
        curr_est_reward = self.r_hat[next_s.idx]
        new_est_reward  = ((curr_est_reward * next_s.visits) + new_reward) / (next_s.visits + 1)

        self.r_hat[next_s.idx] = new_est_reward

    def update_p(self, curr_s, next_s):
        curr_est_p_row   = self.P_hat[curr_s.idx]
        curr_num_of_tran = curr_est_p_row * (curr_s.visits - 1)
        curr_num_of_tran[next_s.idx] += 1

        new_est_p_row      = curr_num_of_tran / curr_s.visits
        self.P_hat[curr_s.idx] = new_est_p_row


class State(object):
    def __init__(self, idx):
        self.idx          = idx
        self.visits       = 0

    def update_visits(self):
        self.visits += 1


class Agent(object):
    def __init__(self, idx, init_state):
        self.idx            = idx
        self.curr_state     = init_state
        self.set_init_state = True

    def update_state(self, new_idx_state):
        self.curr_state = new_idx_state

    def reset_init_state(self):
        self.set_init_state = False


if __name__ == '__main__':
    n = 10
    MDP = MDPModel(n=n)
    for _ in range(100000):
        new_r, new_state = MDP.simulate_one_step(1)

    print(abs(MDP.P - MDP.P_hat).mean())
    print('all done')