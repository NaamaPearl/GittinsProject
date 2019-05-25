from Framework.Plotting import *
import pickle
from Simulator.SimulatorBasics import Runner
import MDPModel.MDPModel as Mdp


def generate_sim_params():
    return {
        'steps': 100, 'eval_type': ['online', 'offline'], 'agents': (10, 30),
        'trajectory_len': 150, 'eval_freq': 50, 'epsilon': 0.15, 'reset_freq': 20000,
        'grades_freq': 50, 'gittins_discount': 0.9, 'temporal_extension': [1], 'T_board': 3, 'runs_per_mdp': 1,
        'varied_param': None, 'trajectory_num': 50, 'max_trajectory_len': 50
    }


def load_mdp_list():
    # clique = pickle.load(open("mdp.pckl", "rb"))
    # directed = pickle.load(open("directed_mdp_with_gittins.pckl", "rb"))
    # clique = pickle.load(open("clique_mdp_with_gittins.pckl", "rb"))
    # cliff = pickle.load(open("cliff_mdp_with_gittins.pckl", "rb"))
    # star = pickle.load(open("star_mdp_with_gittins.pckl", "rb"))
    tunnel = pickle.load(open("tunnel_mdp_with_gittins.pckl", "rb"))

    mdps = [tunnel[0]]
    # mdps = [directed[0], clique[0], cliff[0], tunnel[0]]

    return mdps


def generate_mdp_list():
    def generate_mdp(mdp_type):
        if mdp_type == 'tunnel':
            tunnel_indexes = list(range(n - tunnel_length, n))
            return Mdp.ChainsTunnelMDP(n=n, actions=actions, succ_num=succ_num, op_succ_num=op_succ_num,
                                       chain_num=chain_num,
                                       gamma=gamma, traps_num=0, tunnel_indexes=tunnel_indexes,
                                       reward_param={chain_num - 1: {'bernoulli_p': 1, 'gauss_params': ((10, 4), 0)},
                                                     'lead_to_tunnel': {'bernoulli_p': 1, 'gauss_params': ((-1, 0), 0)},
                                                     'tunnel_end': {'bernoulli_p': 1, 'gauss_params': ((100, 0), 0)}})
        if mdp_type == 'star':
            return Mdp.StarMDP(n=n, actions=actions, succ_num=succ_num, op_succ_num=op_succ_num, chain_num=chain_num,
                               gamma=gamma,
                               reward_param={1: {'bernoulli_p': 1, 'gauss_params': ((100, 3), 0)},
                                             2: {'bernoulli_p': 1, 'gauss_params': ((0, 0), 0)},
                                             3: {'bernoulli_p': 1, 'gauss_params': ((100, 2), 0)},
                                             4: {'bernoulli_p': 1, 'gauss_params': ((1, 0), 0)},
                                             0: {'bernoulli_p': 1, 'gauss_params': ((110, 4), 0)}})
        if mdp_type == 'clique':
            return Mdp.CliquesMDP(n=n, actions=actions, succ_num=succ_num, op_succ_num=op_succ_num, traps_num=0,
                                  chain_num=chain_num, gamma=gamma,
                                  reward_param={chain_num - 1: {'bernoulli_p': 1, 'gauss_params': ((10, 4), 0)}})

        if mdp_type == 'cliff':
            return Mdp.CliffWalker(size=size, random_prob=random_prob, gamma=gamma)
        if mdp_type == 'directed':
            return Mdp.DirectedTreeMDP(depth, actions, gamma, resets_num)

        raise NotImplementedError()
    n = 46
    chain_num = 3
    actions = 3
    succ_num = 3
    op_succ_num = 5
    gamma = 0.95
    tunnel_length = 5
    size = 5
    random_prob = 0.2
    depth = 6
    resets_num = 7

    mdps = [generate_mdp('tunnel')]

    with open('mdp.pckl', 'wb') as f1:
        pickle.dump(mdps, f1)


if __name__ == '__main__':
    ''''build the MDPs or load new'''
    load = False
    mdp_list = load_mdp_list() if load else generate_mdp_list()

    '''define general simulation params. At most 1 parameter can be a list- compare results according to it'''
    gt_compare = False
    general_sim_params = generate_sim_params()

    # _method_dict = {'gittins': ['reward', 'error'], 'greedy': ['reward', 'error'], 'random': [None]}
    # _method_dict = {'gittins': ['reward', 'error'], 'greedy': ['reward', 'error', 'v_f'], 'random': [None]}
    general_sim_params['method_dict'] = {'greedy': ['error', 'reward']}

    if gt_compare:
        general_sim_params['gittins_compare'] = [('model_free', 'error'), ('gittins', 'error')]
        general_sim_params['method_dict']['gittins'].append('ground_truth')

    runner = Runner(general_sim_params, gt_compare=gt_compare, varied_definition_str='temporal_extension')
    res = runner.run(mdp_list)

    with open('run_res2.pckl', 'wb') as f:
        pickle.dump(res, f)

    titles = ['tree']
    if gt_compare:
        plot_results_wrraper('GT', (res, titles))
    else:
        plot_results_wrraper('from main', (res, titles))

    print('all done')
