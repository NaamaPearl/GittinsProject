from Framework.Plotting import *
from Framework import config as cfg
import pickle
from MDPModel import MDPModel as Mdp, MDPUtils as mdpcfg
import argparse
from Framework.Outputs import Runner


def load_mdp_list():
    """load previously generated MDPs"""
    # clique = pickle.load(open("mdp.pckl", "rb"))
    # directed = pickle.load(open("directed_mdp_with_gittins.pckl", "rb"))
    # clique = pickle.load(open("clique_mdp_with_gittins.pckl", "rb"))
    # cliff = pickle.load(open("cliff_mdp_with_gittins.pckl", "rb"))
    # star = pickle.load(open("star_mdp_with_gittins.pckl", "rb"))
    tunnel = pickle.load(open("tunnel_mdp_with_gittins.pckl", "rb"))

    mdps = [tunnel[0]]
    # mdps = [directed[0], clique[0], cliff[0], tunnel[0]]

    return mdps


def generate_mdp_list(type_list):
    """
    Generate new MDP per type in type_list.
    Note that MDPs are generated according to default config. insert values to constructors to override them.
    """
    def generate_mdp(mdp_type):
        if mdp_type == 'tunnel': return Mdp.ChainsTunnelMDP(mdpcfg.TunnelMDPConfig())
        if mdp_type == 'star': return Mdp.StarMDP(mdpcfg.StarMDPConfig())
        if mdp_type == 'clique': return Mdp.CliquesMDP(mdpcfg.CliqueMDPConfig())
        if mdp_type == 'cliff': return Mdp.CliffWalker(mdpcfg.CliffMDPConfig())
        if mdp_type == 'directed': return Mdp.DirectedTreeMDP(mdpcfg.DirectedTreeMDPConfig())

        raise NotImplementedError()

    mdps = [generate_mdp(mdp_type) for mdp_type in type_list]

    with open('mdp.pckl', 'wb') as f1:
        pickle.dump(mdps, f1)

    return mdps


if __name__ == '__main__':
    ''''Parse user's arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', '-l', action='store_true', help='load existing MDP')
    args = parser.parse_args()

    '''define general simulation params. Add arguments to constructor, or use default values'''
    general_sim_params = cfg.SimulationParameters()
    mdp_list = load_mdp_list() if args.load else generate_mdp_list(general_sim_params.mdp_types)

    runner = Runner(general_sim_params)
    res = runner.run(mdp_list)

    with open('run_res2.pckl', 'wb') as f:
        pickle.dump(res, f)

    titles = ['tree']
    if general_sim_params.gt_compare:
        plot_results_wrraper('GT', (res, titles))
    else:
        plot_results_wrraper('from main', (res, titles))

    print('all done')
