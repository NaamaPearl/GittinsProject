import queue as Q
from Actor.Prioritizer import Prioritizer, GittinsPrioritizer, GreedyPrioritizer
from Framework.PrioritizedObject import *
from Critic.Critic import *
from Simulator.SimulatorBasics import *
from Framework.Inputs import *
from collections import Counter


class Simulator:
    def __init__(self, sim_input: ProblemInput):
        self.MDP_model: SimulatedModel = sim_input.MDP_model
        self.evaluation_type = sim_input.eval_type
        self.gamma = sim_input.gamma
        self.epsilon = sim_input.epsilon
        self.evaluated_model = EvaluatedModel()

        self.critic = CriticFactory.Generate(model=self.MDP_model, evaluator_type=self.evaluation_type)
        state_num = self.MDP_model.n
        SimulatedState.action_num = self.MDP_model.actions
        self.evaluated_model.ResetData(self.MDP_model.n, self.MDP_model.actions)
        self.policy = [0] * state_num
        self.MDP_model.CalcPolicyData(self.policy)

        self.InitStatics()

    @property
    def opt_policy(self):
        return self.MDP_model.MDP_model.opt_policy

    def InitStatics(self):
        SimulatedState.policy = self.policy
        SimulatedState.V_hat_vec = self.evaluated_model.V_hat

        StateActionPair.P_hat_mat = self.evaluated_model.P_hat
        StateActionPair.Q_hat_mat = self.evaluated_model.Q_hat
        StateActionPair.TD_error_mat = self.evaluated_model.TD_error
        StateActionPair.r_hat_mat = self.evaluated_model.r_hat
        StateActionPair.visitations_mat = self.evaluated_model.visitations

    def ImprovePolicy(self, sim_input, **kwargs):
        for state in self.MDP_model.states:
            state.policy_action = state.best_action.action

        self.MDP_model.CalcPolicyData(self.policy)

    def Update_V(self, state_action: StateActionPair):
        #  self.V_hat[idx] = self.r_hat[idx] + self.gamma * np.dot(self.P_hat[action][idx, :], self.V_hat)
        state = state_action.state
        state.V_hat = state_action.r_hat + self.gamma * state_action.P_hat @ self.evaluated_model.V_hat

    def Update_Q(self, state_action: StateActionPair, next_state: SimulatedState, reward):
        a_n = (state_action.visitations + 1) ** -0.7
        state_action.TD_error = reward + self.gamma * max(next_state.actions).Q_hat - state_action.Q_hat
        state_action.Q_hat += (a_n * state_action.TD_error)

    def UpdateP(self, state_action: StateActionPair, next_state: SimulatedState):
        curr_num_of_tran = state_action.P_hat * state_action.visitations
        curr_num_of_tran[next_state.idx] += 1

        new_est_p_row = curr_num_of_tran / (state_action.visitations + 1)
        state_action.P_hat = new_est_p_row
        self.UpdatePredecessor(next_state, state_action)

    @staticmethod
    def UpdateReward(state_action: StateActionPair, new_reward):
        state_action.r_hat = (state_action.r_hat * state_action.visitations + new_reward) / (
                state_action.visitations + 1)

    def getActionResults(self, state_action: StateActionPair):
        next_state = self.MDP_model.states[self.MDP_model.GetNextState(state_action)]
        reward = self.MDP_model.GetReward(state_action)
        return next_state, reward

    def updateModel(self, current_state_action, next_state, reward):
        self.UpdateReward(current_state_action, reward)
        self.UpdateP(current_state_action, next_state)
        self.Update_V(current_state_action)
        self.Update_Q(current_state_action, next_state, reward)
        current_state_action.UpdateVisits()

    def SampleStateAction(self, agent_type, state_action: StateActionPair):
        next_state, reward = self.getActionResults(state_action)
        if agent_type == 'regular':
            self.updateModel(state_action, next_state, reward)

        return reward, next_state

    def UpdatePredecessor(self, next_state, state_action):
        pass

    def simulate(self, sim_input):
        for i in range(int(sim_input.steps / sim_input.temporal_extension)):
            self.SimulateOneStep(agents_to_run=sim_input.agents_to_run,
                                 temporal_extension=sim_input.temporal_extension,
                                 iteration_num=i,
                                 T_board=sim_input.T_board)
            if i % sim_input.grades_freq == 0:  # sim_input.grades_freq - 1:
                self.ImprovePolicy(sim_input, iteration_num=i)
            if i % sim_input.evaluate_freq == 0:  # sim_input.evaluate_freq - 1:
                self.SimEvaluate(trajectory_len=sim_input.trajectory_len, running_agents=sim_input.agents_to_run,
                                 gamma=self.gamma)
            if i % sim_input.reset_freq == 0:  # sim_input.reset_freq - 1:
                self.Reset()

        return self.critic

    def Reset(self):
        pass

    def SimulateOneStep(self, agents_to_run, **kwargs):
        pass

    def SimEvaluate(self, **kwargs):
        self.critic.CriticEvaluate(initial_state=self.RaffleInitialState(), good_agents=50,
                                   chain_num=self.MDP_model.MDP_model.chain_num,
                                   active_chains_ratio=self.MDP_model.MDP_model.active_chains_ratio,
                                   active_chains=self.MDP_model.MDP_model.GetActiveChains(), **kwargs)

    def RaffleInitialState(self):
        return np.random.choice(self.MDP_model.states, p=self.MDP_model.MDP_model.init_prob)


class AgentSimulator(Simulator):
    def __init__(self, sim_input: ProblemInput):
        super().__init__(sim_input)
        self.agents_num = sim_input.agent_num
        self.init_prob = self.MDP_model.init_prob
        self.agents = Q.PriorityQueue()
        self.optimal_agents = self.generateOptimalAgents(sim_input.agent_num)
        self.ResetAgents(self.agents_num)

        self.graded_states = {state.idx: random.random() for state in self.MDP_model.states}

    def generateOptimalAgents(self, agents_num):
        agents_list = []
        good_agents = 0
        while good_agents < agents_num:
            new_agent = Agent(100 + good_agents, self.RaffleInitialState(), agent_type='optimal')
            next_state, _ = self.getActionResults(self.ChooseAction(new_agent.curr_state, new_agent.type))
            if next_state.chain not in self.MDP_model.MDP_model.GetActiveChains():
                continue
            new_agent.curr_state = next_state
            agents_list.append(new_agent)
            good_agents += 1

        return agents_list

    def SimEvaluate(self, **kwargs):
        kwargs['agents_reward'] = [agent.object.getOnlineAndZero() for agent in self.agents.queue]
        kwargs['optimal_agents_reward'] = [agent.getOnlineAndZero() for agent in self.optimal_agents]
        super().SimEvaluate(**kwargs)

    def ChooseInitState(self):
        return np.random.choice(self.MDP_model.states, p=self.init_prob)

    def GetStatsForPrioritizer(self, parameter):
        if parameter is None:
            return None, None
        if parameter == 'reward':
            return self.evaluated_model.P_hat, self.evaluated_model.r_hat
        if parameter == 'error':
            return self.evaluated_model.P_hat, abs(self.evaluated_model.TD_error)
        if parameter == 'ground_truth':
            return self.MDP_model.MDP_model.P, np.transpose(self.MDP_model.MDP_model.expected_r)


    def ImprovePolicy(self, sim_input, **kwargs):
        """

        :param sim_input: simulation parameters
        :param kwargs: must contain current iteration number, for reincarnation
        :effect: calculate new indexes for all sates, and grade agents accordingly
        """
        super().ImprovePolicy(sim_input)

        p, r = self.GetStatsForPrioritizer(sim_input.parameter)
        prioritizer = sim_input.prioritizer(states=self.MDP_model.states,
                                            policy=self.policy,
                                            p=p,
                                            r=r,
                                            temporal_extension=sim_input.temporal_extension,
                                            discount_factor=sim_input.gittins_discount)
        self.graded_states = prioritizer.GradeStates()
        self.ReGradeAllAgents(kwargs['iteration_num'], sim_input.grades_freq)

    def ReincarnateAgent(self, agent, iteration_num, grades_freq):
        if iteration_num - agent.last_activation > 3 * grades_freq:
            agent.last_activation = iteration_num
            agent.curr_state = self.RaffleInitialState()

    def ReGradeAllAgents(self, iteration_num, grades_freq):
        """invoked after states re-prioritization. Replaces queue"""
        new_queue = Q.PriorityQueue()
        while self.agents.qsize() > 0:
            agent = self.agents.get().object
            self.ReincarnateAgent(agent, iteration_num, grades_freq)
            new_queue.put(self.GradeAgent(agent))

        self.agents = new_queue

    def GradeAgent(self, agent):
        """ Agents in non-visited states / initial states are prioritized"""
    # if agent.curr_state in self.MDP_model.init_states_idx:
    #     score = -np.inf
    # else:
        score = self.graded_states[agent.curr_state.idx]
        return PrioritizedObject(agent, score)

    def SimulateOneStep(self, agents_to_run, **kwargs):
        """find top-priority agents, and activate them for a single step"""
        agents_list = [self.agents.get().object for _ in range(agents_to_run)]
        self.optimal_agents = self.optimal_agents[:agents_to_run]

        for agent in agents_list + self.optimal_agents:
            for _ in range(kwargs['temporal_extension']):
                self.SimulateAgent(agent, **kwargs)
                if agent.type == 'regular':
                    self.critic.Update(agent.chain, agent.curr_state.idx)

            if agent.type == 'regular':
                self.agents.put(self.GradeAgent(agent))

    def ChooseAction(self, state: SimulatedState, agent_type, T_board=0):
        if agent_type == 'optimal':
            return state.actions[self.opt_policy[state.idx]]

        if agent_type == 'regular':
            min_visits, min_action = state.min_visitations
            if min_visits < T_board:
                return state.actions[min_action]

            return state.policy_action if random.random() > self.epsilon else np.random.choice(state.actions)

    def SimulateAgent(self, agent: Agent, iteration_num, **kwargs):
        """simulate one action of an agent, and re-grade it, according to it's new state"""

        state_action = self.ChooseAction(agent.curr_state, agent.type, kwargs['T_board'])

        agent.last_activation = iteration_num
        reward, next_state = self.SampleStateAction(agent.type, state_action)
        agent.update(reward, next_state)

    def Reset(self):
        self.ResetAgents(self.agents.qsize())

    def ResetAgents(self, agents_num):
        self.agents = Q.PriorityQueue()
        for i in range(agents_num):
            init_state = self.ChooseInitState()
            self.agents.put(PrioritizedObject(Agent(i, init_state), -np.inf))

    @property
    def agents_location(self):
        chains_count = Counter([agent.object.chain for agent in self.agents.queue])
        return chains_count


class PrioritizedSweeping(Simulator):
    def __init__(self, sim_input: ProblemInput):
        self.state_actions_score = np.inf * np.ones((self.MDP_model.n, self.MDP_model.actions))
        self.state_actions = [[PrioritizedObject(state_action, StateActionScore(state_action))
                               for state_action in state.actions] for state in self.MDP_model.states[1:]]
        super().__init__(sim_input)

    def InitStatics(self):
        StateActionScore.score_mat = self.state_actions_score
        super().InitStatics()

    def SimulateOneStep(self, agents_to_run, **kwargs):
        for _ in range(agents_to_run):
            best: PrioritizedObject = max([max(state) for state in self.state_actions])
            best_state_action: StateActionPair = best.object
            self.critic.Update(best_state_action.chain)

            self.SampleStateAction(best_state_action)

            if best_state_action.visitations > kwargs['T_board']:
                best.reward.score = abs(best_state_action.TD_error)

    def UpdateScore(self, state_action_pr: PrioritizedObject):
        state_action = state_action_pr.object

        for predecessor in state_action.state.predecessor.difference(
                {state_action}):  # TODO what if state has probability to stay in the same state
            p = self.evaluated_model.P_hat[predecessor.state.idx][predecessor.action][state_action.state.idx]
            if p == 0:
                raise ValueError('transmission probability should not be 0')
            prioritized_object = self.state_actions[predecessor.state.idx][predecessor.action]
            prioritized_object.reward.score = max(p * abs(state_action.TD_error), prioritized_object.reward.score)
            prioritized_object.active = True

    def UpdatePredecessor(self, next_state, state_action):
        if not (self.MDP_model.MDP_model.type == 'chains' and state_action.state in self.MDP_model.init_states_idx):
            next_state.predecessor.add(state_action)


def SimulatorFactory(mdp: MDPModel, sim_params):
    simulated_mdp = SimulatedModel(mdp)

    return AgentSimulator(
        ProblemInput(MDP_model=simulated_mdp, gamma=mdp.gamma, **sim_params))


def SimInputFactory(method_type, parameter, sim_params):
    simulation_input_type = AgentSimulationInput

    if method_type == 'random':
        parameter = None
        prioritizer = Prioritizer
    elif method_type == 'gittins':
        prioritizer = GittinsPrioritizer
    elif method_type == 'greedy':
        prioritizer = GreedyPrioritizer
    else:
        raise IOError('unrecognized method type:' + method_type)

    return simulation_input_type(prioritizer=prioritizer, parameter=parameter, **sim_params)
