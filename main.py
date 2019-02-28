from Framework.Plotting import *
import pickle
from itertools import product
from functools import reduce


def summarizeCritics(critics, critic_type):
    result = {
        'online': (np.cumsum(np.mean(np.asarray([critic.value_vec['online'] for critic in critics]), axis=0)),
                   np.std(np.asarray([critic.value_vec['online'] for critic in critics]), axis=0)),
        'offline': (np.mean(np.asarray([critic.value_vec['offline'] for critic in critics]), axis=0),
                    np.std(np.asarray([critic.value_vec['offline'] for critic in critics]), axis=0)),
        'bad_states': [np.diff(critic.bad_activated_states) for critic in critics],
        'critics': critics}

    if critic_type in ['chains', 'bridge']:
        result['chain_activations'] = np.mean(np.asarray([critic.chain_activations for critic in critics]), axis=0)

    return result


def RunSimulations(_mdp_list, sim_params):
    sim_definition = reduce(lambda a, b: a + b, [list(product([method], sim_params['method_dict'][method],
                                                              sim_params['temporal_extension']))
                                                 for method in sim_params['method_dict'].keys()])
    result = [(mdp.type, {}, {}, {}) for mdp in _mdp_list]
    for i, mdp in enumerate(_mdp_list):
        print('run MDP num ' + str(i))
        for method, parameter, temp_extension in sim_definition:
            print('     running ' + method + ' prioritization, using ' + str(parameter) + ':')
            sim_input = SimInputFactory(method, parameter, sim_params)
            sim_input.temporal_extension = temp_extension

            critics = []
            indexes = []
            grades = []
            for run_num in range(1, sim_params['runs_per_mdp'] + 1):
                print('         start run # ' + str(run_num))
                critic, index, graded_state = SimulatorFactory(mdp, sim_params).simulate(sim_input)
                critics.append(critic)
                indexes.append(index)
                grades.append(graded_state)
            result[i][1][(method, parameter, temp_extension)] = summarizeCritics(critics, mdp.type)
            result[i][2][(method, parameter, temp_extension)] = indexes
            result[i][3][(method, parameter, temp_extension)] = grades
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

def generateMDP(mdp_type):
    if mdp_type == 'tunnel':
        tunnel_indexes = list(range(n - tunnel_length, n))
        return ChainsTunnelMDP(n=n, actions=actions, succ_num=succ_num, op_succ_num=op_succ_num, chain_num=chain_num,
                               gamma=gamma, traps_num=0, tunnel_indexes=tunnel_indexes,
                               reward_param={chain_num - 1: {'bernoulli_p': 1, 'gauss_params': ((10, 4), 0)},
                                             'lead_to_tunnel': {'bernoulli_p': 1, 'gauss_params': ((-1, 0), 0)},
                                             'tunnel_end': {'bernoulli_p': 1, 'gauss_params': ((100, 0), 0)}})
    if mdp_type == 'star':
        return StarMDP(n=n, actions=actions, succ_num=succ_num, op_succ_num=op_succ_num, chain_num=chain_num,
                       gamma=gamma,
                       reward_param={1: {'bernoulli_p': 1, 'gauss_params': ((100, 3), 0)},
                                     2: {'bernoulli_p': 1, 'gauss_params': ((0, 0), 0)},
                                     3: {'bernoulli_p': 1, 'gauss_params': ((100, 2), 0)},
                                     4: {'bernoulli_p': 1, 'gauss_params': ((1, 0), 0)},
                                     0: {'bernoulli_p': 1, 'gauss_params': ((110, 4), 0)}})
    if mdp_type == 'cliques':
        return SeperateChainsMDP(n=n, actions=actions, succ_num=succ_num, op_succ_num=op_succ_num, traps_num=0,
                                 chain_num=chain_num, gamma=gamma,
                                 reward_param={chain_num - 1: {'bernoulli_p': 1, 'gauss_params': ((10, 4), 0)}})
    if mdp_type == 'gittins':
        return GittinsMDP(n=n, actions=actions, succ_num=succ_num, op_succ_num=op_succ_num, chain_num=chain_num,
                          gamma=gamma, terminal_probability=0.05,
                          reward_param={1: {'first': {'gauss_params': ((100, 3), 0)}},
                                        2: {'bernoulli_p': 1, 'gauss_params': ((0, 0), 0)},
                                        3: {'bernoulli_p': 1, 'gauss_params': ((100, 2), 0)},
                                        4: {'bernoulli_p': 1, 'gauss_params': ((1, 0), 0)},
                                        0: {'bernoulli_p': 1, 'gauss_params': ((110, 4), 0)}})

    if mdp_type == 'cliff':
        return CliffWalker(size=size,  random_prob=random_prob, gamma=gamma)
    if mdp_type == 'directed':
        return DirectedTreeMDP(depth, actions, gamma, resets_num)

    raise NotImplementedError()


def EvaluateGittins(res_list, general_sim_params):
    pickle.load(open("run_res2.pckl", "rb"))
    mdp_num = len(res_list)
    subs = []
    for i, mdp_res in enumerate(res_list):
        evaluated_indexes = np.asarray(res[i][2][('gittins', 'reward', 1)][0])
        gt_indexes = np.asarray(res[i][2][('gittins', 'ground_truth', 1)][0])
        eval_count = int(general_sim_params['steps'] /
                         (general_sim_params['eval_freq']))
        max_step = eval_count * general_sim_params['eval_freq']
        steps = np.linspace(0, max_step, num=eval_count)
        sub = np.abs(evaluated_indexes - gt_indexes)
        for j in range(sub.shape[1]):
            if gt_indexes[0][j] > (10 ** (-1)):
                sub[j] /= np.abs(gt_indexes[0][j])
        sub = sub.sum(1)
        state_num = evaluated_indexes[0].shape[0]
        subs.append(sub / state_num)
    subs = np.asarray(subs).T
    plt.figure()
    plt.plot(steps, subs)
    plt.xlabel('simulation steps')
    plt.ylabel(r'$\frac{1}{N}\sum|I_{(s)}-\tilde{I}_{(s)}|$')
    plt.title('Gittins Calculation in Evaluated Model')
    plt.legend(['Tree', 'Clique', 'Cliff', 'Star', 'Tunnel'])
    plt.show()

if __name__ == '__main__':
    # building the MDPs
    load = True
    if load:
        directed = pickle.load(open("directed_mdp.pckl", "rb"))
        clique = pickle.load(open("clique_mdp.pckl", "rb"))
        cliff = pickle.load(open("cliff_mdp.pckl", "rb"))
        star = pickle.load(open("star_mdp.pckl", "rb"))
        tunnel = pickle.load(open("tunnel_mdp.pckl", "rb"))

        mdp_list = [directed[0]]
        # mdp_list = [directed[0], clique[0], cliff[0], star[0], tunnel[0]]

    else:
        n = 46
        chain_num = 3
        actions = 3
        succ_num = 3
        op_succ_num = 5
        gamma = 0.9
        tunnel_length = 5
        size = 5
        random_prob = 0.2
        depth = 6
        resets_num = 7

        mdp_list = [generateMDP('directed')]

        with open('mdp.pckl', 'wb') as f:
            pickle.dump(mdp_list, f)

    # define general simulation params
    general_sim_params = {
        'steps': 5000, 'eval_type': ['online', 'offline'], 'agents_to_run': 10, 'agents_to_generate': 30,
        'trajectory_len': 150, 'eval_freq': 50, 'epsilon': 0.15, 'reset_freq': 10000,
        'grades_freq': 50, 'gittins_discount': 0.9, 'temporal_extension': [1], 'T_board': 3, 'runs_per_mdp': 5
    }
    opt_policy_reward = [mdp.CalcOptExpectedReward() for mdp in mdp_list]

    # _method_dict = {'gittins': ['reward', 'error'], 'greedy': ['reward', 'error'], 'random': [None]}
    _method_dict = {'gittins': ['reward', 'error','ground_truth'], 'greedy': ['reward', 'error','ground_truth']}
    general_sim_params['method_dict'] = _method_dict

    res = RunSimulations(mdp_list, sim_params=general_sim_params)

    printalbe_res = {'res': res, 'opt_reward': opt_policy_reward, 'params': general_sim_params}

    with open('run_res2.pckl', 'wb') as f:
        pickle.dump(printalbe_res, f)

    # EvaluateGittins(res, general_sim_params)
    PlotResults(res, opt_policy_reward, general_sim_params)

    print('all done')
