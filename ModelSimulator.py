import numpy as np
import copy
import queue as Q
from Prioritizer import Prioritizer, GittinsPrioritizer
import random
from MDPModel import MDPModel, Agent, PrioritizedObject


class Simulator:
    def __init__(self, MDP_model, randomprio=True, agent_num=5, init_state=0):
        self.MDP_model = copy.deepcopy(MDP_model)
        if randomprio:
            self.prioritizer = Prioritizer(self.MDP_model)
        else:
            self.prioritizer = GittinsPrioritizer(self.MDP_model)
        self.graded_states = {state.idx: random.random() for state in self.MDP_model.s}
        self.agents = Q.PriorityQueue()
        [self.agents.put(PrioritizedObject(Agent(i, init_state), i)) for i in range(agent_num)]  # TODO - Random init_state

    def ApproxModel(self):
        self.graded_states = self.prioritizer.GradeStates()
        self.ReGradeAllAgents()

    # invoked after states re-prioritization. Replaces queue
    def ReGradeAllAgents(self):
        new_queue = Q.PriorityQueue()
        while self.agents.qsize() > 0:
            new_queue.put(self.GradeAgent(self.agents.get().object))

        self.agents = new_queue

    def GradeAgent(self, agent):
        return PrioritizedObject(agent, self.graded_states[agent.curr_state])

    def simulate(self, steps=10000, grades_freq=20):
        for i in range(steps):
            self.SimulateOneStep()
            if i % grades_freq == grades_freq - 1:
                self.ApproxModel()  # prioritize agents & states

    # find top-priority agents, and activate them for a single step
    def SimulateOneStep(self, agents_to_run=1):
        agents_list = [self.agents.get().object for _ in range(agents_to_run)]
        for agent in agents_list:
            self.SimulateAgent(agent)

    # simulate one action of an agent, and re-grade it, according to it's new state
    def SimulateAgent(self, agent):
        curr_s = self.MDP_model.s[agent.curr_state]
        next_s = self.choose_next_state(curr_s)
        r = self.MDP_model.r[curr_s.idx]

        self.MDP_model.update_reward(curr_s, r)
        self.MDP_model.update_p(curr_s, next_s)
        self.MDP_model.update_V(curr_s.idx)
        agent.curr_state = next_s.idx
        curr_s.update_visits()

        self.agents.put(self.GradeAgent(agent))

    def choose_next_state(self, curr_s):
        next_s_idx = np.random.choice(np.arange(self.MDP_model.n), p=self.MDP_model.P[curr_s.idx])
        return self.MDP_model.s[next_s_idx]

    def evaluate_P_hat(self):
        return self.MDP_model.P_hat_sum_diff()

    def EvaluateV(self):
        return self.MDP_model.V_hat_diff()


if __name__ == '__main__':
    n = 10
    k = 4

    MDP = MDPModel(n=n)
    RandomSimulator  = Simulator(MDP, randomprio=True)
    GittinsSimulator = Simulator(MDP, randomprio=False)

    RandomSimulator.simulate(steps=100000)
    GittinsSimulator.simulate(steps=10000)

    print('eval Random')
    print(RandomSimulator.evaluate_P_hat())
    print(RandomSimulator.EvaluateV())

    print('eval Gittin')
    print(GittinsSimulator.evaluate_P_hat())
    print(GittinsSimulator.EvaluateV())

    # print(GittinsSimulator.evaluateGittins())
    print('all done')
