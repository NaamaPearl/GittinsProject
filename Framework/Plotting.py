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




def CompareActivations(data_output, mdp_i):
    chain_num = len(next(iter(data_output.values())).get('chain_activations'))
    plt.figure()
    tick_shift = np.linspace(-0.35, 0.35, len(data_output))
    for _iter, key in enumerate(data_output):
        ticks = [tick_shift[_iter] + s for s in range(chain_num)]
        plt.bar(ticks, data_output[key]['chain_activations'], width=0.1, align='center',
                label=str(key[0]) + ' ' + str(key[1]))

    plt.xticks(range(chain_num), ['chain ' + str(s) for s in range(chain_num)])
    plt.title('Agents Activation per Chains for mdp num ' + str(mdp_i))
    plt.legend()


def PlotEvaluation(data_output, optimal_policy_reward, general_sim_params):
    params = ['all']
    # params = ['reward', 'error', 'all']
    [PlotEvaluationForParam(data_output, optimal_policy_reward, param, general_sim_params)
     for param in params]


def PlotColor(method, param=None):
    if method == 'optimal':
        return '0' # black

    if method == 'random':
        return 'xkcd:lime green'

    ## Error
    if param == 'error':
        if method == 'gittins':
            return 'xkcd:red'
        if method == 'greedy':
            return 'xkcd:orange'

    ## Reward
    if param == 'reward':
        if method == 'gittins':
            return 'xkcd:bright blue'
        if method == 'greedy':
            return 'xkcd:sky blue'


def CreateZoomFig(ax):
    # sub region of the original image
    axins = zoomed_inset_axes(ax, 5, loc=7)
    offsetx = 1000
    offsety = 17000
    x1, x2, y1, y2 = 7800 + offsetx, 8200 + offsetx, 12000 + offsety, 19000 + offsety
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    # plt.axis('off')
    # axins.Axes.xtick(visible=False)
    # plt.yticks(visible=False)

    return axins


def CreateLegendFig():
    fig = pylab.figure(figsize=(3, 2))
    plt.axis('off')
    return fig


def BuildLegend():
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0][::-1]))
    labels = list(labels)
    handles = list(handles)
    rand_idx = labels.index('random None') if 'random None' in labels else None
    if rand_idx is not None:
        labels.remove('random None')
        labels.append('random')
        handles.append(handles.pop(rand_idx))

    by_label = OrderedDict(zip(labels, handles))
    leg = plt.figlegend(by_label.values(), by_label.keys(), ncol=len(labels), loc=8)

    for line in leg.get_lines():
        line.set_linewidth(4.0)


def SetDefaults():
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    mscale.register_scale(CustomScale)
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE, titleweight="bold")  # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE, labelweight="bold")  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE, titleweight="bold")  # fontsize of the figure title


def CalcData(general_sim_params, sim_outputs, method, parameter, temp_ext, eval_type):
    eval_count = int(general_sim_params['steps'] /
                     (general_sim_params['eval_freq'] * temp_ext))
    max_step = eval_count * general_sim_params['eval_freq'] * temp_ext
    samples = np.linspace(0, max_step, num=eval_count)
    steps = np.linspace(0, max_step, num=eval_count * temp_ext)

    mean_values, std_tmp = sim_outputs[(method, parameter, temp_ext)].get(eval_type)
    mean_values_smooth = np.array(smooth(mean_values))

    y = np.interp(steps, samples, mean_values_smooth)
    std = np.interp(steps, samples, std_tmp)

    return y, std, steps


def PlotRegret(sim_outputs, req_param, general_sim_params, temporal_extension_run):
    ax = plt.Subplot(global_fig, inner[inner_j])
    global_fig.add_subplot(ax)
    axes[outer_i, inner_j] = ax
    axins = CreateZoomFig(ax)

    for method, parameter, temp_ext in sim_outputs.keys():
        if parameter == req_param or req_param == 'all':

            y, std, steps = CalcData(general_sim_params, sim_outputs, method, parameter, temp_ext, 'online')

            if not temporal_extension_run:
                ax.plot(steps, y, color=PlotColor(method, parameter), label=method + ' ' + str(parameter))
                axins.plot(steps, y, color=PlotColor(method, parameter), label=method + ' ' + str(parameter))
            else:
                ax.plot(steps, y, color=PlotColor(method, parameter), label=r'$\lambda$ = ' + str(temp_ext))
                axins.plot(steps, y, color=PlotColor(method, parameter),
                                      label=r'$\lambda$ = ' + str(temp_ext))
            ax.fill_between(steps, y + std / 4, y - std / 4, alpha=0.5, color=PlotColor(method, parameter))
            axins.fill_between(steps, y + std / 4, y - std / 4, alpha=0.5,
                                          color=PlotColor(method, parameter))

    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5")


def PlotOffline(sim_outputs, req_param, general_sim_params, temporal_extension_run, optimal_policy_reward):
    ax = plt.Subplot(global_fig, inner[inner_j])
    global_fig.add_subplot(ax)
    axes[outer_i, inner_j] = ax

    for method, parameter, temp_ext in sim_outputs.keys():
        if parameter == req_param or req_param == 'all':

            y, std, steps = CalcData(general_sim_params, sim_outputs, method, parameter, temp_ext, 'offline')

            if not temporal_extension_run:
                ax.plot(steps, y, color=PlotColor(method, parameter), label=method + ' ' + str(parameter))
            else:
                ax.plot(steps, y, color=PlotColor(method, parameter), label=r'$\lambda$ = ' + str(temp_ext))
            ax.fill_between(steps, y + std / 4, y - std / 4, alpha=0.5, color=PlotColor(method, parameter))

    # ax.set_yscale('custom')

    ax.axhline(y=optimal_policy_reward, color=PlotColor('optimal'), linestyle='-',
                  label='optimal policy expected reward')


def PlotEvaluationForParam(sim_outputs, optimal_policy_reward, req_param, general_sim_params):
    SetDefaults()
    try:
        temporal_extension_run = len(general_sim_params['temporal_extension']) > 1
    except:
        temporal_extension_run = False

    if outer_i == 0:
        PlotRegret(sim_outputs, req_param, general_sim_params, temporal_extension_run)
    if outer_i == 1:
        PlotOffline(sim_outputs, req_param, general_sim_params, temporal_extension_run, optimal_policy_reward)


def PlotResults(result_list, opt_policy_reward_list, general_sim_params):
    for i, ((mdp_type, res_data), opt_reward) in enumerate(zip(result_list, opt_policy_reward_list)):
        PlotEvaluation(res_data, opt_reward, general_sim_params)
        # if mdp_type in ['chains', 'bridge']:
            # CompareActivations(res_data, i)

        # plt.show()



global_fig = None
MAIN_FOLDER = r'C:\Users\Naama\Dropbox\project\report graphs\\'
def DATA_PATH(path):
    return MAIN_FOLDER + path
if __name__ == '__main__':
    graph_name = [r'cliques\TD and Reward\5 actions\run_res2.pckl',
                  r'star\3 actions\run_res2.pckl',
                  r'star\3 actions\run_res2.pckl',
                  r'cliques\TD and Reward\3 actions\run_res2.pckl',
                  r'grid\run_res2.pckl']
    data_path = [DATA_PATH(name) for name in graph_name]

    global_fig = plt.figure(figsize=(20, 16))
    outer = gridspec.GridSpec(2, 1, wspace=0.3, hspace=0.3)

    axes = np.empty(shape=(2, 5), dtype=object)
    for outer_i in [0,1]:
        inner = gridspec.GridSpecFromSubplotSpec(1, 5,
                                                 subplot_spec=outer[outer_i], wspace=0.2, hspace=0.3)
        for inner_j, path in enumerate(data_path):
            res_tuple = pickle.load(open(path, 'rb'))
            PlotResults(res_tuple['res'], res_tuple['opt_reward'], res_tuple['params'])

    axes[0,0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axes[1,2].set_xlabel('simulation_steps')

    axes[0,0].set_ylabel('evaluated reward')
    axes[1,0].set_ylabel('average reward')

    axes[0,2].set_title('Regret')
    axes[1,2].set_title('Evaluation')
    plt.suptitle('Reward Evaluation \naverage of ' + str(3) + ' runs') ## TODO
    BuildLegend()
    global_fig.show()


