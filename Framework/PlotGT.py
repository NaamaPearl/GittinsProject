import pickle
import matplotlib.pyplot as plt
from matplotlib import scale as mscale

from Framework.CustomScale import CustomScale
from Framework.plotUtils import *


def calc_data(general_sim_params, sim_outputs, optimal, param1, param2):
    eval_count = int(general_sim_params.steps /
                     (general_sim_params.eval_freq))
    max_step = eval_count * general_sim_params.eval_freq
    steps = np.linspace(0, max_step, num=eval_count)

    gittins_mean_values, gittins_std_tmp = sim_outputs.result[('gittins', param1, 1)]['offline']
    gittins_smooth = np.array(smooth(gittins_mean_values))
    gt_mean_values, gt_std_tmp = sim_outputs.result[('gittins', param2, 1)]['offline']
    gt_smooth = np.array(smooth(gt_mean_values))

    return steps, (gittins_smooth, gittins_std_tmp), (
        (gt_smooth - gittins_smooth) / optimal, (gt_std_tmp - gittins_std_tmp) / optimal ** 2)


# Global
global_dict = {}
global_fig = None
MAIN_FOLDER = r'C:\Users\yonio\PycharmProjects\GittinsProject\\'


def list_of_mdp_from_pckl():
    # part
    # titles = ['Cliques', 'Tunnel', 'Tree', 'Cliff']
    titles = ['Cliques']
    graph_name = [r'run_res2.pckl'
                  # r'GT/new calc/GT_clique.pckl',
                  # r'GT/new calc/GT_tunnel.pckl',
                  # r'GT/new calc/GT_tree.pckl',
                  # r'GT/new calc/GT_cliff.pckl'
                  ]
    data_path = [DATA_PATH(name) for name in graph_name]
    mdp_num = len(graph_name)

    res_tuple_list = {'res': [], 'opt_reward': []}
    for i, path in enumerate(data_path):
        res_tuple = pickle.load(open(path, 'rb'))
        res_tuple_list['res'].append(res_tuple['res'][0])
        res_tuple_list['opt_reward'].append(res_tuple['opt_reward'][0])
    res_tuple_list['params'] = res_tuple['params']

    return res_tuple_list, titles, mdp_num


def DATA_PATH(path):
    return MAIN_FOLDER + path


def evaluate_gittins_by_value(res_list, titles):
    subs = []
    method_list = res_list[0].sim_params.gittins_compare
    for (method, param) in method_list:
        for i, mdp_res in enumerate(res_list):
            general_sim_params = mdp_res.sim_params
            eval_count = int(general_sim_params.steps / (general_sim_params.eval_freq))
            max_step = eval_count * general_sim_params.eval_freq
            steps = np.linspace(0, max_step, num=eval_count)

            evaluated_indexes = np.asarray(mdp_res.result[(method, param, 1)]['indices']['eval'][i])
            gt_indexes = np.asarray(mdp_res.result[(method, param, 1)]['indices']['gt'][i])
            sub = np.abs(evaluated_indexes - gt_indexes) / mdp_res.optimal_reward
            sub = sub.sum(1)
            state_num = evaluated_indexes[0].shape[0]
            subs.append(sub / state_num)
    subs = np.asarray(subs).T
    global_dict['axes'][0].plot(steps, subs)
    # axes[0].set_xlabel('simulation steps')
    global_dict['axes'][0].set_ylabel(r'$\frac{1}{N}\sum\frac{|I_{(s)}-\tilde{I}_{(s)}|}{MDP_{optimal reward}}$',
                                      fontsize=15)
    global_dict['axes'][0].set_title('Gittins Index Differences')
    # global_dict['axes'][0].set_ylim(0, 0.02)
    global_dict['axes'][0].legend([x if x != 'reward' else 'model_based' for x in method_list])


def evaluate_gittins_by_states(res_list, titles):
    method_list = res_list[0].sim_params.gittins_compare
    for (method, param) in method_list:
        bad_states = []
        for i, mdp_res in enumerate(res_list):
            general_sim_params = mdp_res.sim_params
            bad_states.append(
                np.asarray(mdp_res.result[(method, param, 1)]['bad_states'][i]) / general_sim_params.eval_freq)
            eval_count = int(general_sim_params.steps /
                             (general_sim_params.eval_freq))
            max_step = eval_count * general_sim_params.eval_freq
            steps = np.linspace(0, max_step, num=eval_count)[:-1]
        bad_states = np.asarray(bad_states).T / 10
        global_dict['axes'][1].plot(steps, bad_states)
    global_dict['axes'][1].set_xlabel('simulation steps')
    global_dict['axes'][1].set_ylabel('wrong decisions')
    # global_dict['axes'][1].set_ylim(-0.02, 0.02)
    global_dict['axes'][1].set_xlim(0, 500)
    vals = global_dict['axes'][1].get_yticks()
    global_dict['axes'][1].set_yticklabels(['{:.0%}'.format(x) for x in vals])
    global_dict['axes'][1].set_title('Accuracy of Agents Chosen')
    global_dict['axes'][1].legend([x if x != 'reward' else 'model_based' for x in method_list])


def evaluate_gittins_by_perf(res_list, titles):
    method_list = res_list[0].sim_params.method_dict['gittins']
    param2 = 'ground_truth'
    for i, mdp_res in enumerate(res_list):
        # for param1, param2 in set(itertools.combinations(method_list, 2)):  # FOR COMPARISION
        general_sim_params = mdp_res.sim_params
        for param1 in method_list:
            steps, (y, std), (y_diff, std_diff) = calc_data(general_sim_params, mdp_res, mdp_res.optimal_reward, param1, param2)

            # c = PlotColor(method, parameter)
            global_dict['axes'][2].plot(steps, y, label=param1 if param1 != 'reward' else 'model_based')

            # global_dict['axes'][2].plot(steps, y, label=param1 + ' vs ' + param2)
            # global_dict['axes'][2].fill_between(steps, y + std, y - std, alpha=0.2)

        global_dict['axes'][2].axhline(y=mdp_res.optimal_reward, color='0', linestyle='-',
                                       label='optimal expected reward')

    # axes[2].set_xlabel('simulation steps')
    global_dict['axes'][2].set_ylabel(r'$V_{approximated}$', fontsize=15)
    global_dict['axes'][2].set_title('Performance Difference')
    global_dict['axes'][2].legend()


def evaluate_gittins(res_list, titles):
    evaluate_gittins_by_value(res_list, titles)
    evaluate_gittins_by_states(res_list, titles)
    evaluate_gittins_by_perf(res_list, titles)


def set_defaults():
    SMALL_SIZE = 13
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    mscale.register_scale(CustomScale)
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE, titleweight="bold")  # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE, labelweight="bold")  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE, titleweight="bold")  # fontsize of the figure title


def plot_gt(results=None):
    if results is None:
        res_tuple_list, titles, _ = list_of_mdp_from_pckl()
    else:
        res_tuple_list, titles = results
    set_defaults()
    fig, global_dict['axes'] = plt.subplots(1, 3)
    fig.subplots_adjust(wspace=0.5, bottom=0.2)
    evaluate_gittins(res_tuple_list, titles)

    plt.show()


if __name__ == '__main__':
    plot_gt()
