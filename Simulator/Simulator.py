import queue as Q
from Actor.Prioritizer import Prioritizer, GittinsPrioritizer, GreedyPrioritizer
from Framework.PrioritizedObject import *
from Critic.Critic import *
from Simulator.SimulatorBasics import *
from Framework.Inputs import *
from collections import Counter


class Simulator:
    def __init__(self, sim_input: ProblemInput, name):
        self.MDP_model: SimulatedModel = sim_input.MDP_model
        self.evaluation_type = sim_input.eval_type
        self.gamma = sim_input.gamma
        self.epsilon = sim_input.epsilon
        self.evaluated_model = EvaluatedModel()
        self.policy = None
        self.critic = None
        self.name = name

        self.InitParams(eval_type=self.evaluation_type)

    def InitStatics(self):
        SimulatedState.policy = self.policy
        SimulatedState.V_hat_vec = self.evaluated_model.V_hat

        StateActionPair.P_hat_mat = self.evaluated_model.P_hat
        StateActionPair.Q_hat_mat = self.evaluated_model.Q_hat
        StateActionPair.TD_error_mat = self.evaluated_model.TD_error
        StateActionPair.r_hat_mat = self.evaluated_model.r_hat
        StateActionPair.visitations_mat = self.evaluated_model.visitations

    def InitParams(self, **kwargs):
        self.critic = CriticFactory.Generate(model=self.MDP_model,
                                             evaluator_type=kwargs['eval_type'])
        state_num = self.MDP_model.n
        SimulatedState.action_num = self.MDP_model.actions
        self.evaluated_model.ResetData(self.MDP_model.n, self.MDP_model.actions)
        self.policy = [random.randint(0, self.MDP_model.actions - 1) for _ in range(state_num)]
        self.MDP_model.CalcPolicyData(self.policy)

        self.InitStatics()

    def ImprovePolicy(self, sim_input, step):
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

    def SampleStateAction(self, state_action: StateActionPair):
        next_state = self.MDP_model.states[self.MDP_model.GetNextState(state_action)]
        reward = self.MDP_model.GetReward(state_action)
        self.UpdateReward(state_action, reward)
        self.UpdateP(state_action, next_state)

        self.Update_V(state_action)
        self.Update_Q(state_action, next_state, reward)
        state_action.UpdateVisits()

        return reward, next_state

    def UpdateP(self, state_action: StateActionPair, next_state: SimulatedState):
        curr_num_of_tran = state_action.P_hat * state_action.visitations
        curr_num_of_tran[next_state.idx] += 1

        new_est_p_row = curr_num_of_tran / (state_action.visitations + 1)
        state_action.P_hat = new_est_p_row
        self.UpdatePredecessor(next_state, state_action)

    def UpdatePredecessor(self, next_state, state_action):
        pass

    @staticmethod
    def UpdateReward(state_action: StateActionPair, new_reward):
        state_action.r_hat = (state_action.r_hat * state_action.visitations + new_reward) / (
                state_action.visitations + 1)

    def simulate(self, sim_input):
        self.InitParams(eval_type=self.evaluation_type)

        for i in range(sim_input.steps):
            self.SimulateOneStep(agents_to_run=sim_input.agents_to_run, T_board=sim_input.T_bored, iteration_num=i)
            if i % sim_input.grades_freq == sim_input.grades_freq - 1:
                self.ImprovePolicy(sim_input, i)
            if i % sim_input.evaluate_freq == sim_input.evaluate_freq - 1:
                self.SimEvaluate(trajectory_len=sim_input.trajectory_len, running_agents=sim_input.agents_to_run,
                                 gamma=self.gamma)
            if i % sim_input.reset_freq == sim_input.reset_freq - 1:
                self.Reset()

    def Reset(self):
        pass

    def SimulateOneStep(self, agents_to_run, **kwargs):
        pass

    def SimEvaluate(self, **kwargs):
        try:
            self.critic.CriticEvaluate(initial_state=self.RaffleInitialState(), good_agents=50,
                                       chain_num=self.MDP_model.MDP_model.chain_num,
                                       active_chains=self.MDP_model.MDP_model.active_chains, **kwargs)
        except AttributeError:
            self.critic.CriticEvaluate(initial_state=self.RaffleInitialState(), good_agents=50,
                                       chain_num=self.MDP_model.MDP_model.chain_num, **kwargs)

    def RaffleInitialState(self):
        return np.random.choice(self.MDP_model.states, p=self.MDP_model.MDP_model.init_prob)


class AgentSimulator(Simulator):
    def __init__(self, sim_input: ProblemInput, name):
        self.agents_num = sim_input.agent_num
        super().__init__(sim_input, name)
        self.agents = Q.PriorityQueue()
        self.graded_states = None
        self.init_prob = None

    def SimEvaluate(self, **kwargs):
        kwargs['agents_reward'] = [agent.object.accumulated_reward for agent in self.agents.queue]
        # kwargs['running_agents'] = min(
        #     reduce((lambda x, y: x + y), (agent.object.chain for agent in self.agents.queue)),
        #     kwargs['running_agents'])
        super().SimEvaluate(**kwargs)

    def InitParams(self, **kwargs):
        super().InitParams(**kwargs)
        self.graded_states = {state.idx: random.random() for state in self.MDP_model.states}
        self.init_prob = self.MDP_model.init_prob
        self.ResetAgents(self.agents_num)
        self.init_prob = self.MDP_model.init_prob

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
            reward_mat = list(zip(*self.MDP_model.MDP_model.expected_r))
            return self.MDP_model.MDP_model.P, reward_mat

    def ImprovePolicy(self, sim_input, iteration_num):
        super().ImprovePolicy(sim_input, iteration_num)

        p, r = self.GetStatsForPrioritizer(sim_input.parameter)
        self.graded_states = sim_input.prioritizer.GradeStates(states=self.MDP_model.states,
                                                               policy=self.policy,
                                                               p=p,
                                                               r=r,
                                                               look_ahead=sim_input.gittins_look_ahead,
                                                               discount=sim_input.gittins_discount,
                                                               visits=np.min(self.evaluated_model.visitations, axis=1),
                                                               T_bored=sim_input.T_bored)
        self.ReGradeAllAgents(sim_input.T_bored, iteration_num)

    def ReincarnateAgent(self, agent, iteration_num):
        if iteration_num - agent.last_activation > 30 and self.name != "random":
            agent.last_activation = iteration_num
            agent.curr_state = self.RaffleInitialState()

        return agent

    def ReGradeAllAgents(self, T_board, iteration_num):
        """invoked after states re-prioritization. Replaces queue"""
        new_queue = Q.PriorityQueue()
        while self.agents.qsize() > 0:
            agent = self.agents.get().object
            self.ReincarnateAgent(agent, iteration_num)
            new_queue.put(self.GradeAgent(agent, T_board))

        self.agents = new_queue

    def GradeAgent(self, agent, T_board):
        """ Agents in non-visited states / initial states are prioritized"""
        top_priority = agent.curr_state.visitations < T_board or agent.chain is None
        score = -np.inf if top_priority else self.graded_states[agent.curr_state.idx]

        return PrioritizedObject(agent, score)

    def SimulateOneStep(self, agents_to_run, **kwargs):
        """find top-priority agents, and activate them for a single step"""
        agents_list = [self.agents.get().object for _ in range(agents_to_run)]
        for agent in agents_list:
            self.critic.Update(agent.chain)
            self.SimulateAgent(agent, kwargs['T_board'], kwargs['iteration_num'])
            self.agents.put(self.GradeAgent(agent, kwargs['T_board']))

    def ChooseAction(self, state: SimulatedState, T_board):
        min_visits, min_action = state.min_visitations
        if min_visits < T_board:
            return state.actions[min_action]
        if random.random() < self.epsilon:
            return np.random.choice(state.actions)

        return state.policy_action

    def SimulateAgent(self, agent: Agent, T_board, iteration_num):
        """simulate one action of an agent, and re-grade it, according to it's new state"""
        agent.last_activation = iteration_num

        state_action = self.ChooseAction(agent.curr_state, T_board)

        reward, next_state = self.SampleStateAction(state_action)
        agent.accumulated_reward += reward
        agent.curr_state = next_state

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
    def __init__(self, sim_input: ProblemInput, name):
        self.state_actions = None
        self.state_actions_score = None
        super().__init__(sim_input, name)

    def InitParams(self, **kwargs):
        self.state_actions_score = np.inf * np.ones((self.MDP_model.n, self.MDP_model.actions))
        super().InitParams(**kwargs)
        self.state_actions = [[PrioritizedObject(state_action, StateActionScore(state_action))
                               for state_action in state.actions] for state in self.MDP_model.states[1:]]

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


def SimulatorFactory(mdp: MDPModel, name, sim_params):
    simulated_mdp = SimulatedModel(mdp)
    agent_num = sim_params['agents_to_run'] * sim_params['agents_ratio']

    return AgentSimulator(
        ProblemInput(MDP_model=simulated_mdp, agent_num=agent_num, gamma=mdp.gamma, name=name, **sim_params), name)


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

    return simulation_input_type(prioritizer=prioritizer(), parameter=parameter, **sim_params)
