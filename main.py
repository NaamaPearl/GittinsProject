from Framework.Plotting import *
import pickle
from itertools import product
from functools import reduce


def summarizeCritics(critics, critic_type):
    result = {'online': (np.mean(np.asarray([critic.value_vec['online'] for critic in critics]), axis=0),
                         np.std(np.asarray([critic.value_vec['online'] for critic in critics]), axis=0)),
              'offline': (np.mean(np.asarray([critic.value_vec['offline'] for critic in critics]), axis=0),
                          np.std(np.asarray([critic.value_vec['offline'] for critic in critics]), axis=0))}

    if critic_type == 'chains':
        result['chain_activations'] = np.mean(np.asarray([critic.chain_activations for critic in critics]), axis=0)

    return result


def RunSimulations(_mdp_list, sim_params):
    sim_definition = reduce(lambda a, b: a + b, [list(product([method], sim_params['method_dict'][method]))
                                                 for method in sim_params['method_dict'].keys()])
    result = [(mdp.type, {}) for mdp in _mdp_list]
    for i, mdp in enumerate(_mdp_list):
        print('run MDP num ' + str(i))
        for method, parameter in sim_definition:
            print('     running ' + method + ' prioritization, using ' + str(parameter))
            sim_input = SimInputFactory(method, parameter, sim_params)
            result[i][1][(method, parameter)] = summarizeCritics([SimulatorFactory(mdp, sim_params).simulate(sim_input)
                                                                  for _ in range(sim_params['runs_per_mdp'])], mdp.type)
    return result


# def compareSweepingWithAgents(mdp, sim_params, agent_ratio_vec):
#     general_sim_params['eval_type'] = ['offline']
#     sweeper = PrioritizedSweeping(ProblemInput(
#         MDP_model=SimulatedModel(mdp), agent_num=sim_params['agents_to_run'], gamma=mdp.gamma, **sim_params),
#         'sweeping')
#     sweeper.simulate(SimulationInput(**sim_params))
#     sweeping_result = sweeper.critic.value_vec['offline']
#
#     agents_result = []
#     for agent_ratio in agent_ratio_vec:
#         sim_params['agent_ratio'] = agent_ratio
#
#         agent_simulator = SimulatorFactory(mdp, 'gittins', sim_params)
#         agent_simulator.simulate(SimInputFactory('greedy', 'error', sim_params))
#
#         agents_result.append(agent_simulator.critic.value_vec['offline'])
#
#     plt.figure()
#     eval_count = int(np.ceil(general_sim_params['steps'] / general_sim_params['eval_freq']))
#     steps = np.array(list(range(eval_count))) * general_sim_params['eval_freq']
#
#     plt.plot(steps, sweeping_result, label='sweeping')
#     for i in range(len(agents_result)):
#         plt.plot(steps, agents_result[i]['offline'], label=r'$\rho$ = ' + str(agent_ratio_vec[i]))
#
#     plt.legend()
#     plt.show()


# def compareLookAhead(mdp, sim_params, look_ahead_vec, optimal_reward):
#     simulator = SimulatorFactory(mdp, sim_params)
#
#     # param_list = ['error', 'reward']
#     param_list = ['reward']
#     for param in param_list:
#         outputs = {look_ahead: ChainSimulationOutput(sim_params['eval_type']) for look_ahead in look_ahead_vec}
#         for look_ahead in look_ahead_vec:
#             for i in range(sim_params['runs_per_mdp']):
#                 sim_params['look_ahead'] = look_ahead
#                 simulator.simulate(SimInputFactory('gittins', param, sim_params))
#                 outputs[look_ahead].reward_eval.add(simulator.critic.value_vec)
#
#         PlotLookAhead(outputs, param, optimal_reward, general_sim_params)
#
#     plt.show()


if __name__ == '__main__':
    # building the MDPs
    tunnel_length = 5
    load = False
    if load:
        mdp_list = pickle.load(open("mdp.pckl", "wb"))
    else:
        star_mdp = StarMDP(n=46, actions=5, succ_num=3, op_succ_num=5, chain_num=5, gamma=0.9,
                           reward_param={1: {'bernoulli_p': 1, 'gauss_params': ((100, 3), 2)},
                                         2: {'bernoulli_p': 1, 'gauss_params': ((0, 0), 0)},
                                         3: {'bernoulli_p': 1, 'gauss_params': ((50, 2), 2)},
                                         4: {'bernoulli_p': 1, 'gauss_params': ((1, 0), 0)},
                                         0: {'bernoulli_p': 1, 'gauss_params': ((87, 3), 2)}})

        mdp_list = [star_mdp]

        with open('mdp.pckl', 'wb') as f:
            pickle.dump(mdp_list, f)

    # define general simulation params
    general_sim_params = {
        'steps': 5000, 'eval_type': ['online', 'offline'], 'agents_to_run': 15, 'agents_ratio': 3,
        'trajectory_len': 150, 'eval_freq': 50, 'epsilon': 0.15, 'reset_freq': 10000,
        'grades_freq': 50, 'gittins_discount': 0.9, 'temporal_extension': 1, 'T_board': 3, 'runs_per_mdp': 3
    }
    opt_policy_reward = [mdp.CalcOptExpectedReward(general_sim_params) for mdp in mdp_list]

    _method_dict = {'gittins': ['reward', 'error'], 'greedy': ['reward', 'error'], 'random': [None]}
    general_sim_params['method_dict'] = _method_dict

    res = RunSimulations(mdp_list, sim_params=general_sim_params)

    printalbe_res = {'res': res, 'opt_reward': opt_policy_reward, 'params': general_sim_params}

    with open('run_res2.pckl', 'wb') as f:
        pickle.dump(printalbe_res, f)

    PlotResults(res, opt_policy_reward, general_sim_params)

    print('all done')
