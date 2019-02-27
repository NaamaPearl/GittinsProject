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


def PlotColor(method, param=None, l=None):
    if l is not None:
        # c = {1: 'green', 2: 'blue', 3: 'purple', 4: 'pink', 8: 'light blue', 15:'aqua'}
        c = {1: 'pale blue', 2: 'baby blue', 3: 'baby blue', 4: 'cerulean', 8: 'blue', 15:'indigo'}
        return 'xkcd:' + c[l]
        # return (0,0,c[l])

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

    if param == 'ground_truth':
        if method == 'gittins':
            return 'xkcd:ochre'
        if method == 'greedy':
            return 'xkcd:pale yellow'

def CreateZoomFig(ax):
    # sub region of the original image
    zoom = zoom_list[inner_j]
    loc = loc_list[inner_j]
    axins = zoomed_inset_axes(ax, zoom, loc=loc)
    # offsetx = 1000
    # offsety = 17000
    # x1, x2, y1, y2 = 7800 + offsetx, 8200 + offsetx, 12000 + offsety, 19000 + offsety
    offset = offset_list[outer_i][inner_j]
    if offset is not None:
        axins.set_xlim(offset[0], offset[1])
        axins.set_ylim(offset[2], offset[3])
    else:
        axins.set_visible(False)
    plt.yticks([], visible=False)
    plt.xticks([], visible=False)

    return axins


def CreateLegendFig():
    fig = pylab.figure(figsize=(3, 2))
    plt.axis('off')
    return fig


def BuildLegend(TE=False):
    handles, labels = plt.gca().get_legend_handles_labels()
    if not TE:
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


def CalcData(general_sim_params, sim_outputs, method, parameter, temp_ext, eval_type):
    eval_count = int(general_sim_params['steps'] /
                     (general_sim_params['eval_freq'] * temp_ext))
    max_step = eval_count * general_sim_params['eval_freq'] * temp_ext
    samples = np.linspace(0, max_step, num=eval_count)
    steps = np.linspace(0, max_step, num=eval_count * temp_ext)

    mean_values, std_tmp = sim_outputs[(method, parameter, temp_ext)].get(eval_type)
    mean_values_smooth = np.array(smooth(mean_values))
    if samples.shape < mean_values_smooth.shape:
        pad_len = mean_values_smooth.shape[0]-samples.shape[0]
        samples = np.pad(samples, (0,pad_len),'edge')
    y = np.interp(steps, samples, mean_values_smooth)
    std = np.interp(steps, samples, std_tmp)

    return y[:100], std[:100], steps[:100]

def NeedToPlot(req_param, param, method):
    if req_param == 'GT':
        if method not in ['greedy']:
            return True

    if req_param == 'all':
        return True

    if param in req_param:
        return True
    return False

def PlotRegret(ax, sim_outputs, req_param, general_sim_params, temporal_extension_run):

    axins = CreateZoomFig(ax)

    for method, parameter, temp_ext in sim_outputs.keys():
        if NeedToPlot(req_param, parameter, method):

            y, std, steps = CalcData(general_sim_params, sim_outputs, method, parameter, temp_ext, 'online')

            if not temporal_extension_run:
                c = PlotColor(method, parameter)
                ax.plot(steps, y, color=c, label=method + ' ' + str(parameter))
                axins.plot(steps, y, color=c, label=method + ' ' + str(parameter))
            else:
                c = PlotColor(method, parameter, temp_ext)
                ax.plot(steps, y, color=c, label=r'$\lambda$ = ' + str(temp_ext))
                axins.plot(steps, y, color=c,
                                      label=r'$\lambda$ = ' + str(temp_ext))
            ax.fill_between(steps, y + std / 4, y - std / 4, alpha=0.5, color=c)
            axins.fill_between(steps, y + std / 4, y - std / 4, alpha=0.5,
                                          color=c)

    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    offset = offset_list[outer_i][inner_j]
    line = line_loc[inner_j]
    if offset is not None:
        mark_inset(ax, axins, loc1=line[0], loc2=line[1], fc=(1,1,1), ec="0.5")


def PlotOffline(ax, sim_outputs, req_param, general_sim_params, temporal_extension_run, optimal_policy_reward):
    for method, parameter, temp_ext in sim_outputs.keys():

        if NeedToPlot(req_param, parameter, method):

            y, std, steps = CalcData(general_sim_params, sim_outputs, method, parameter, temp_ext, 'offline')

            if not temporal_extension_run:
                c = PlotColor(method, parameter)
                ax.plot(steps, y, color=c, label=method + ' ' + str(parameter))
            else:
                c = PlotColor(method, parameter, temp_ext)
                ax.plot(steps, y, color=c, label=r'$\lambda$ = ' + str(temp_ext))

            ax.fill_between(steps, y + std / 4, y - std / 4, alpha=0.1, color=c)

    # ax.set_yscale('custom')

    ax.axhline(y=optimal_policy_reward, color=PlotColor('optimal'), linestyle='-',
                  label='optimal policy expected reward')

    ax.set_ylim([ylim1[inner_j] * optimal_policy_reward, ylim2[inner_j] * optimal_policy_reward])


def PlotEvaluationForParam(sim_outputs, optimal_policy_reward, req_param, general_sim_params):
    SetDefaults()
    try:
        temporal_extension_run = len(general_sim_params['temporal_extension']) > 1
    except:
        temporal_extension_run = False


    ax = plt.Subplot(global_fig, inner[inner_j])
    global_fig.add_subplot(ax)
    axes[outer_i, inner_j] = ax
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 3))

    if outer_i == 0:
        PlotRegret(ax, sim_outputs, req_param, general_sim_params, temporal_extension_run)
    if outer_i == 1:
        PlotOffline(ax, sim_outputs, req_param, general_sim_params, temporal_extension_run, optimal_policy_reward)


def PlotResults(result_list, opt_policy_reward_list, general_sim_params):
    for i, ((mdp_type, res_data), opt_reward) in enumerate(zip(result_list, opt_policy_reward_list)):
        PlotEvaluation(res_data, opt_reward, general_sim_params)
        # if mdp_type in ['chains', 'bridge']:
            # CompareActivations(res_data, i)

        # plt.show()

global_fig = None
MAIN_FOLDER = r'C:\Users\Naama\Dropbox\project\report graphs\\'

def ListOfMDPFromPckl():
    # part
    ylim1 = [0.9, 0.95, 0.7, 0.7,   0.2, 0.7]
    ylim2 = [1.01, 1.005, 1.03,  1.03, 1.08, 1.03]
    # x1, x2, y1, y2
    offset_list = [[(8300, 8500, 8500, 10000),
                    (8000, 9000, 15000, 17500),
                    (7200, 7500, 28000, 32000),
                    (8000, 9000, 20000, 30000),
                    (8000, 9000, 300, 320),
                    (8000, 9000, 20000, 30000)],
                   [None, None, None, None, None, None]]
    zoom_list = [10, 3, 7, 2, 3, 3]
    loc_list = [8, 4, 4, 4, 4, 4]
    line_loc = [(4, 1), (1, 2), (1, 3), (2, 1), (1, 2), (3, 1)]
    titles = ['Cliques', 'Star', 'Tunnel', 'Tree', 'Cliff']
    graph_name = [r'cliques\TD and Reward\5 actions\run_res2.pckl',
                  r'star\3 actions\run_res2.pckl',
                  r'tunnel\run_res2_withTD.pckl',
                  r'run_res2.pckl',
                  r'clif\run_res2.pckl'

                  ]
    data_path = [DATA_PATH(name) for name in graph_name]
    mdp_num = len(graph_name)

    res_tuple_list = {'res': [], 'opt_reward': []}
    for i, path in enumerate(data_path):
        res_tuple = pickle.load(open(path, 'rb'))
        res_tuple_list['res'].append(res_tuple['res'][0])
        res_tuple_list['opt_reward'].append(res_tuple['opt_reward'][0])
    res_tuple_list['params'] = res_tuple['params']

    return res_tuple_list, titles, zoom_list, loc_list, ylim1, ylim2, offset_list, line_loc, mdp_num


def ListOfMDPFromPath():
    ylim1 = [0.95, 0.87, 0.5, 0.95, 0.7]
    ylim2 = [1.001, 1.03, 1.005, 1.03, 1.03]
    titles = ['Tree', 'Cliques', 'Cliff', 'Star', 'Tunnel']
    res_tuple_list = pickle.load(open(r'C:\Users\Naama\Dropbox\project\report graphs\6_mdps_res_1.pckl', 'rb'))
    # x1, x2, y1, y2
    offset_list = [[(8300, 8500, 8500, 10000),
                    (8000, 9000, 300, 320),
                    (8000, 9000, 15000, 17500),
                    (7200, 7500, 28000, 32000),
                    (8000, 9000, 20000, 30000),
                    (8000, 9000, 20000, 30000)],
                   [None, None, None, None, None, None]]
    zoom_list = [10, 3, 3, 7, 2, 3]
    loc_list = [8, 4, 4, 4, 4, 4]
    line_loc = [(4, 1), (1, 2), (1, 2), (1, 3), (2, 1), (3, 1)]
    mdp_num = len(res_tuple_list['res'])
    return res_tuple_list, titles, zoom_list, loc_list, ylim1, ylim2, offset_list, line_loc, mdp_num

def DATA_PATH(path):
    return MAIN_FOLDER + path


def FormatPlot(mdp_num):
    if mdp_num > 1:
        axes[1, 2].set_xlabel('simulation_steps')

        axes[0, 0].set_ylabel('evaluated regret')
        axes[1, 0].set_ylabel('average reward')

        for i, ax in enumerate(axes[0]):
            ax.set_title(titles[i])
        axes[0, 2].set_title('Regret' + '\n\n' + titles[2])
        axes[1, 2].set_title('Evaluation')
    else:
        axes[1,0].set_xlabel('simulation_steps')
        axes[0, 0].set_ylabel('evaluated regret')
        axes[1, 0].set_ylabel('average reward')
        axes[0, 0].set_title('Regret')
        axes[1, 0].set_title('Evaluation')

    plt.suptitle('Reward Evaluation')

    BuildLegend(mdp_num)
    global_fig.show()

def GTRes():
    res_tuple_list = pickle.load(open(DATA_PATH(r'GT\GT_clique.pckl'), 'rb'))

    ylim1 = [0.7]
    ylim2 = [1.001]
    titles = ['']
    # x1, x2, y1, y2
    offset_list = [[None],[None]]
    zoom_list = [10]
    loc_list = [8]
    line_loc = [(4, 1)]
    mdp_num = len(res_tuple_list['res'])
    return res_tuple_list, titles, zoom_list, loc_list, ylim1, ylim2, offset_list, line_loc, mdp_num


def TERes():
    res_tuple_add = [pickle.load(open(DATA_PATH(r'temporal_extension\run_res_TE_8.pckl'), 'rb')),
                     pickle.load(open(DATA_PATH(r'temporal_extension\run_res_TE_16.pckl'), 'rb'))]
    res_tuple_list = pickle.load(open(DATA_PATH(r'temporal_extension\run_res2.pckl'), 'rb'))

    keys = []
    values = []
    for res_tuple in res_tuple_add:
            [keys.append(key) for key in res_tuple['res'][0][1].keys()]
            [values.append(value) for value in res_tuple['res'][0][1].values()]

    for key, value in zip(keys, values):
        res_tuple_list['res'][0][1][key] = value

    del res_tuple_list['res'][0][1][('gittins', 'reward', 2)]

    ylim1 = [0.5]
    ylim2 = [1.001]
    titles = ['']
    # x1, x2, y1, y2
    offset_list = [[None],[None]]
    zoom_list = [10]
    loc_list = [8]
    line_loc = [(4, 1)]
    mdp_num = len(res_tuple_list['res'])
    return res_tuple_list, titles, zoom_list, loc_list, ylim1, ylim2, offset_list, line_loc, mdp_num


if __name__ == '__main__':

    res_tuple_list, titles, zoom_list, loc_list, ylim1, ylim2, offset_list, line_loc, mdp_num = TERes()

    global_fig = plt.figure(figsize=(10, 8))
    outer = gridspec.GridSpec(2, 1, wspace=0.3, hspace=0.3)

    axes = np.empty(shape=(2, mdp_num), dtype=object)
    for outer_i in [0, 1]:
        inner = gridspec.GridSpecFromSubplotSpec(1, mdp_num,
                                                 subplot_spec=outer[outer_i], wspace=0.3, hspace=0.3)
        for inner_j in range(mdp_num):
            PlotEvaluation(res_tuple_list['res'][inner_j][1], res_tuple_list['opt_reward'][inner_j],
                           res_tuple_list['params'])

    FormatPlot(True)



