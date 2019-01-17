from __future__ import division
import pickle
import matplotlib.pyplot as plt
import numpy as np
from Framework.Plotting import smooth
# import networkx as nx
from MDPModel.MDPModel import ChainsTunnelMDP
# import pydot


def PlotLookAhead(results_dict, param, optimal_reward, general_sim_params):
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.suptitle(param + ' Based Gittins Prioritization \nlook ahead comparison')

    eval_count = int(np.ceil(general_sim_params['steps'] / general_sim_params['eval_freq']))
    steps = np.array(list(range(eval_count))) * general_sim_params['eval_freq']

    for i, eval_type in enumerate(['online', 'offline']):
        for look_ahead in results_dict.keys():
            result = results_dict[look_ahead]
            reward_eval = result.reward_eval.get(eval_type)
            smoothed_eval = np.array([smooth(reward_eval[i])[:-10] for i in range(reward_eval.shape[0])])

            y = np.mean(smoothed_eval, axis=0)
            std = np.std(smoothed_eval, axis=0)
            ax[i].plot(steps, y, label=r'$\lambda$ = ' + str(look_ahead))
            ax[i].fill_between(steps, y + std / 2, y - std / 2, alpha=0.5)

        ax[i].set_xlabel('simulation steps')
        ax[i].set_ylabel('evaluated reward')
        ax[i].set_title(eval_type)
        ax[i].legend()

    ax[1].axhline(y=optimal_reward, color='r', linestyle='-', label='optimal policy expected reward')

# def PlotPMatrix(mdp):
#     states = list(zip(range(mdp.n), mdp.opt_r))
#     P = list(mdp.opt_P)
#
#     G = nx.MultiDiGraph()
#     labels = {}
#     edge_labels = {}
#
#     for i, origin_state in enumerate(states):
#         for j, destination_state in enumerate(states):
#             rate = P[i][j]
#             if rate > 0:
#                 G.add_edge(origin_state, destination_state, weight=rate, label="{:.02f}".format(rate))
#                 edge_labels[(origin_state, destination_state)] = label = "{:.02f}".format(rate)
#
#     plt.figure(figsize=(10, 7))
#     node_size = 200
#     pos = {state: list(state) for state in states}
#     nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
#     nx.draw_networkx_labels(G, pos, font_weight=2)
#     nx.draw_networkx_edge_labels(G, pos, edge_labels)
#     plt.axis('off')
#     # plt.savefig("../images/mc-matplotlib.svg", bbox_inches='tight')
#
#     nx.drawing.nx_pydot.write_dot(G, 'mc.dot')


if __name__ == '__main__':
    tunnel_length = 2
    _mdp_list = [ChainsTunnelMDP(n=16, actions=4, succ_num=2, op_succ_num=4, chain_num=3, gamma=0.99, traps_num=0,
                                 tunnel_indexes=list(range(13, 13 + tunnel_length)),
                                 reward_param={2: {'bernoulli_p': 1, 'gauss_params': ((10, 1), 0)},
                                               'lead_to_tunnel': {'bernoulli_p': 1, 'gauss_params': ((-1, 0), 0)},
                                               'tunnel_end': {'bernoulli_p': 1, 'gauss_params': ((100, 0), 0)}})]
    PlotPMatrix(_mdp_list[0])

    # with open('temporal_results1.pickle', 'rb') as handle:
    #     temporal_results_list = pickle.load(handle)
    #
    # reward_result = []
    # error_result = []
    # for temporal_result in temporal_results_list:
    #     reward_result.append(temporal_result[0]['gittins']['reward'])
    #     error_result.append(temporal_result[0]['gittins']['error'])
    #
    # PlotLookAhead(reward_result, 'Reward')
    # PlotLookAhead(error_result, 'Error')
    #
    print('all done')
