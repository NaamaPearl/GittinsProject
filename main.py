from Framework.Plotting import *
import pickle
from itertools import product
from Simulator.Simulator import *


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


def RunSimulations(_mdp_list, sim_params, varied_definition_str, gt_compare=False):
    result = [{'type': mdp.type, 'critics': {}, 'indices': {}} for mdp in _mdp_list]

    sim_definition = reduce(lambda a, b: a + b, [list(product([method], sim_params['method_dict'][method],
                                                              sim_params[varied_definition_str]))
                                                 for method in sim_params['method_dict'].keys()])

    for i, mdp in enumerate(_mdp_list):
        print('run MDP num ' + str(i))
        for method, parameter, varied_definition in sim_definition:
            curr_sim_result = result[i]
            res_key = method, parameter, varied_definition
            curr_sim_result['indices'][res_key] = {'eval': [], 'gt': []}
            critic_list = []

            print('     running ' + method + ' prioritization, using ' + str(parameter) +
                  ', with ' + varied_definition_str + ' = ' + str(varied_definition) + ':')
            sim_params[varied_definition_str] = varied_definition
            sim_input = SimInputFactory(method, parameter, sim_params)

            for run_num in range(1, sim_params['runs_per_mdp'] + 1):
                print('         start run # ' + str(run_num))
                sim = SimulatorFactory(mdp, sim_params, gt_compare)
                sim_result = sim.simulate(sim_input)
                if gt_compare:
                    critic, index, gt = sim_result
                    curr_sim_result['indices'][res_key]['eval'].append(index)
                    curr_sim_result['indices'][res_key]['gt'].append(gt)
                else:
                    critic = sim_result
                critic_list.append(critic)

            curr_sim_result['critics'][res_key] = summarizeCritics(critic_list, mdp.type)

    return result


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
        return CliquesMDP(n=n, actions=actions, succ_num=succ_num, op_succ_num=op_succ_num, traps_num=0,
                          chain_num=chain_num, gamma=gamma,
                          reward_param={chain_num - 1: {'bernoulli_p': 1, 'gauss_params': ((10, 4), 0)}})

    if mdp_type == 'cliff':
        return CliffWalker(size=size,  random_prob=random_prob, gamma=gamma)
    if mdp_type == 'directed':
        return DirectedTreeMDP(depth, actions, gamma, resets_num)

    raise NotImplementedError()


if __name__ == '__main__':
    # building the MDPs
    load = True
    if load:
        # clique = pickle.load(open("mdp.pckl", "rb"))
        # directed = pickle.load(open("directed_mdp_with_gittins.pckl", "rb"))
        # clique = pickle.load(open("clique_mdp_with_gittins.pckl", "rb"))
        # cliff = pickle.load(open("cliff_mdp_with_gittins.pckl", "rb"))
        # star = pickle.load(open("star_mdp_with_gittins.pckl", "rb"))
        tunnel = pickle.load(open("tunnel_mdp_with_gittins.pckl", "rb"))

        mdp_list = [tunnel[0]]
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

        mdp_list = [generateMDP('tunnel')]

        with open('mdp.pckl', 'wb') as f:
            pickle.dump(mdp_list, f)

    # define general simulation params. At most 1 parameter can be a list- compare results according to it
    general_sim_params = {
        'steps': 1000, 'eval_type': ['online', 'offline'], 'agents': (10, 30),
        'trajectory_len': 150, 'eval_freq': 50, 'epsilon': 0.15, 'reset_freq': 10000,
        'grades_freq': 50, 'gittins_discount': 0.9, 'temporal_extension': [1], 'T_board': 3, 'runs_per_mdp': 1,
        'varied_param': None, 'trajectory_num': 2, 'max_trajectory_len': 2
    }
    opt_policy_reward = [mdp.CalcOptExpectedReward() for mdp in mdp_list]

    gt_comapre = False

    # _method_dict = {'gittins': ['reward', 'error'], 'greedy': ['reward', 'error'], 'random': [None]}
    _method_dict = {'gittins': ['model_free', 'reward', 'ground_truth']}  # 'greedy': ['reward', 'error','ground_truth']}
    general_sim_params['method_dict'] = _method_dict

    if gt_comapre:
        general_sim_params['gittins_compare'] = ['model_free', 'reward']
        general_sim_params['method_dict']['gittins'].append('ground_truth')
    res = RunSimulations(mdp_list, general_sim_params, varied_definition_str='temporal_extension',
                         gt_compare=gt_comapre)

    printable_res = {'res': res, 'opt_reward': opt_policy_reward, 'params': general_sim_params}

    with open('run_res2.pckl', 'wb') as f:
        pickle.dump(printable_res, f)

    titles = ['tree']
    if gt_comapre:
        PlotResultsWrraper('GT', (printable_res, titles))
    else:
        PlotResultsWrraper('from main', (printable_res, titles))

    print('all done')
