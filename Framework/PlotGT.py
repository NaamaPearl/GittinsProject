import matplotlib.pyplot as plt
from Simulator.Simulator import *
import pickle
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import pylab
from Framework.CustomScale import CustomScale
from Framework.plotUtils import *
from matplotlib import scale as mscale
import matplotlib.gridspec as gridspec
from collections import OrderedDict
from matplotlib.ticker import FormatStrFormatter


def CalcData(general_sim_params, sim_outputs, optimal):
    eval_count = int(general_sim_params['steps'] /
                     (general_sim_params['eval_freq']))
    max_step = eval_count * general_sim_params['eval_freq']
    steps = np.linspace(0, max_step, num=eval_count)

    gittins_mean_values, gittins_std_tmp = sim_outputs[1][('gittins', 'error', 1)].get('offline')
    gittins_smooth = np.array(smooth(gittins_mean_values))
    gt_mean_values, gt_std_tmp = sim_outputs[1][('gittins', 'ground_truth', 1)].get('offline')
    gt_smooth = np.array(smooth(gt_mean_values))

    return (gt_smooth - gittins_smooth) / optimal, steps, (gt_std_tmp - gittins_std_tmp) / optimal ** 2






global_fig = None
MAIN_FOLDER = r'C:\Users\Naama\Dropbox\project\report graphs\\'

def ListOfMDPFromPckl():
    # part
    titles = ['Cliques', 'Tunnel', 'Tree', 'Cliff']
    graph_name = [r'GT/new calc/GT_clique.pckl',
                 r'GT/new calc/GT_tunnel.pckl',
                 r'GT/new calc/GT_tree.pckl',
                 r'GT/new calc/GT_cliff.pckl'
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


def EvaluateGittinsByValue(res_list, general_sim_params, titles, optimal):
    subs = []
    for i, mdp_res in enumerate(res_list):

        evaluated_indexes = np.asarray(mdp_res[2][('gittins', 'reward', 1)]['eval'])[4]
        gt_indexes = np.asarray(mdp_res[2][('gittins', 'ground_truth', 1)]['gt'])
        eval_count = int(general_sim_params['steps'] /
                         (general_sim_params['eval_freq']))
        max_step = eval_count * general_sim_params['eval_freq']
        steps = np.linspace(0, max_step, num=eval_count)
        sub = np.abs(evaluated_indexes - gt_indexes) / optimal[i]
        # for j in range(sub.shape[1]):
        #     if gt_indexes[0][j] != 0:
        #         sub[j] /= np.abs(gt_indexes[0][j])
        sub = sub.sum(1)
        state_num = evaluated_indexes[0].shape[0]
        subs.append(sub / state_num)
    subs = np.asarray(subs).T
    axes[0].plot(steps, subs)
    # axes[0].set_xlabel('simulation steps')
    axes[0].set_ylabel(r'$\frac{1}{N}\sum\frac{|I_{(s)}-\tilde{I}_{(s)}|}{MDP_{optimal reward}}$', fontsize=15)
    axes[0].set_title('Gittins Index Differences')
    axes[0].legend(titles)


def EvaluateGittinsByStates(res_list, general_sim_params, titles):
    bad_states = []
    for i, mdp_res in enumerate(res_list):

        bad_states.append(np.asarray(mdp_res[1][('gittins', 'reward', 1)]['bad_states'][0]) / general_sim_params['eval_freq'])
        eval_count = int(general_sim_params['steps'] /
                         (general_sim_params['eval_freq']))
        max_step = eval_count * general_sim_params['eval_freq']
        steps = np.linspace(0, max_step, num=eval_count)[:-1]
    bad_states = np.asarray(bad_states).T / 10
    axes[1].plot(steps, bad_states)
    axes[1].set_xlabel('simulation steps')
    axes[1].set_ylabel('wrong decisions')
    vals = axes[1].get_yticks()
    axes[1].set_yticklabels(['{:.0%}'.format(x) for x in vals])
    axes[1].set_title('Accuracy of Agents Chosen')
    axes[1].legend(titles)


def EvaluateGittinsByPerf(res_list, general_sim_params, titles, optimal):
    for i, mdp_res in enumerate(res_list):
        y, steps, std = CalcData(general_sim_params, mdp_res, optimal[i])

        # c = PlotColor(method, parameter)
        axes[2].plot(steps, y)
        axes[2].fill_between(steps, y + std, y - std, alpha=0.2)

    # axes[2].set_xlabel('simulation steps')
    axes[2].set_ylabel(r'$\frac{V_{ground\ truth} - V_{approximated}}{MDP_{optimal reward}}$', fontsize=15)
    axes[2].set_title('Performance Difference')
    axes[2].legend(titles)

def EvaluateGittins(res_list, general_sim_params, titles):
     # mdp_num = len(res_list)
    EvaluateGittinsByValue(res_list['res'], general_sim_params, titles, res_tuple_list['opt_reward'])
    EvaluateGittinsByStates(res_list['res'], general_sim_params, titles)
    EvaluateGittinsByPerf(res_list['res'], general_sim_params, titles, res_tuple_list['opt_reward'])



def SetDefaults():
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




if __name__ == '__main__':
    res_tuple_list, titles, mdp_num = ListOfMDPFromPckl()
    SetDefaults()
    fig, axes = plt.subplots(1, 3)
    fig.subplots_adjust(wspace=0.5, bottom = 0.2 )
    EvaluateGittins(res_tuple_list, res_tuple_list['params'], titles)

    plt.show()

