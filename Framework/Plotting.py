import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import pylab
from Framework.CustomScale import CustomScale
from Framework.plotUtils import *
from matplotlib import scale as mscale
import matplotlib.gridspec as gridspec
from collections import OrderedDict
from Framework.PlotGT import PlotGT

# Global
global_dict = {}


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


def PlotColor(method, param=None, varied_param_str=None, l=None):
    if varied_param_str == 'temporal_extension':
        if method in 'random':
            #     c = {1: 'yellow green', 2: 'baby blue', 3: 'toxic green', 4: 'dark pastel green', 8: 'dark grass green', 15: 'spruce'}
            return 'xkcd:yellow green'
        # c = {1: 'green', 2: 'blue', 3: 'purple', 4: 'pink', 8: 'light blue', 15:'aqua'}
        c = {1: 'robin\'s egg blue', 2: 'baby blue', 3: 'baby blue', 4: 'cerulean', 8: 'blue', 15: 'indigo'}
        return 'xkcd:' + c[l]
        # return (0,0,c[l])

    if varied_param_str == 'agents':
        c = {(10, 10): 'robin\'s egg blue',
             (10, 20): 'baby blue',
             (10, 30): 'blue',
             (10, 40): 'indigo',

             (10, 40): 'robin\'s egg blue',
             (20, 40): 'baby blue',
             (30, 40): 'blue',
             (40, 40): 'indigo'
        }
        return 'xkcd:' + c[l]

    if method == 'optimal':
        return '0'  # black

    if method == 'random':
        return 'xkcd:lime green'

        ## Error
    if param == 'error':
        if method == 'gittins':
            return 'xkcd:red'
        if method == 'greedy':
            return 'xkcd:orange'
        if method == 'model_free':
            return 'xkcd:yellow'

    ## v_f
    if param == 'v_f':
        return  'xkcd:purple'

    ## Reward
    if param == 'reward':
        if method == 'gittins':
            return 'xkcd:bright blue'
        if method == 'greedy':
            return 'xkcd:sky blue'
        if method == 'model_free':
            return 'xkcd:robin\'s egg blue'

    if param == 'ground_truth':
        if method == 'gittins':
            return 'xkcd:ochre'
        if method == 'greedy':
            return 'xkcd:pale yellow'


def CreateZoomFig(ax, optimal_policy_reward):
    # sub region of the original image
    zoom = global_dict['zoom_list'][global_dict['j']]
    loc = global_dict['loc_list'][global_dict['j']]
    axins = zoomed_inset_axes(ax, zoom, loc=loc)
    offset = global_dict['offset_list'][global_dict['i']][global_dict['j']]
    if offset is not None:
        axins.set_xlim(offset[0], offset[1])
        if global_dict['i'] == 1:
            axins.set_ylim(offset[2], offset[3])
        else:
            axins.set_ylim(offset[2] / optimal_policy_reward, offset[3] / optimal_policy_reward)
    else:
        axins.set_visible(False)
    plt.yticks([], visible=False)
    plt.xticks([], visible=False)

    return axins


def CreateLegendFig():
    fig = pylab.figure(figsize=(3, 2))
    plt.axis('off')
    return fig


def BuildLegend(mdp_num, varied_param=None):

    if varied_param == 'agents':
        for ax in global_dict['axes'][1]:
            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            by_label['optimal'] = by_label['optimal policy expected reward']
            del by_label['optimal policy expected reward']
            # leg = plt.figlegend(by_label.values(), by_label.keys(), ncol=len(by_label), loc=8)
            leg = ax.legend(by_label.values(), by_label.keys())
        return

    handles, labels = global_dict['axes'][1,0].get_legend_handles_labels()
    if varied_param != 'temporal_extension':
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0][::-1]))
        labels = list(labels)
        handles = list(handles)
        rand_idx = labels.index('random None') if 'random None' in labels else None
        if rand_idx is not None:
            labels.remove('random None')
            labels.append('random')
            handles.append(handles.pop(rand_idx))
    by_label = OrderedDict(zip(labels, handles))

    if mdp_num > 1:
        leg = plt.figlegend(by_label.values(), by_label.keys(), ncol=min(len(by_label), 5), loc=8)
    else:
        by_label['optimal'] = by_label['optimal policy expected reward']
        del by_label['optimal policy expected reward']
        leg = global_dict['axes'][1, 0].legend(by_label.values(), by_label.keys())

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


def CalcData(general_sim_params, sim_output, varied_param, eval_type, optimal_policy_reward):
    if general_sim_params['varied_param'] == 'temporal_extension':
        temp_ext = varied_param
    else:
        temp_ext = general_sim_params['temporal_extension']

    eval_count = int(general_sim_params['steps'] /
                     (general_sim_params['eval_freq'] * temp_ext))
    max_step = eval_count * general_sim_params['eval_freq'] * temp_ext
    samples = np.linspace(0, max_step, num=eval_count)
    steps = np.linspace(0, max_step, num=eval_count * temp_ext)

    mean_values, std_tmp = sim_output.get(eval_type)
    mean_values_smooth = np.array(smooth(mean_values))
    if samples.shape < mean_values_smooth.shape:
        pad_len = mean_values_smooth.shape[0] - samples.shape[0]
        samples = np.pad(samples, (0, pad_len), 'edge')
    y = np.interp(steps, samples, mean_values_smooth)
    std = np.interp(steps, samples, std_tmp)

    return y / optimal_policy_reward, std / optimal_policy_reward, steps


def NeedToPlot(req_param, param, method, TE=False, l=0):
    if TE and l != 1 and method == 'random':
        return False
    if req_param == 'GT':
        if method not in ['greedy']:
            return True

    if req_param == 'all':
        return True

    if param in req_param:
        return True
    return False


def PlotData(ax, sim_outputs, req_param, general_sim_params, optimal_policy_reward, eval_type):
    axins = CreateZoomFig(ax, optimal_policy_reward)

    for method, parameter, varied_param in sim_outputs.keys():
        label = createLabel(general_sim_params['varied_param'], varied_param, method, parameter)
        if NeedToPlot(req_param, parameter, method):
            y, std, steps = CalcData(general_sim_params,
                                     sim_outputs[(method, parameter, varied_param)],
                                     varied_param,
                                     eval_type,
                                     optimal_policy_reward)

            c = PlotColor(method, parameter, general_sim_params['varied_param'], varied_param)
            ax.plot(steps, y, color=c, label=label)
            axins.plot(steps, y, color=c, label=label)

            ax.fill_between(steps, y + std / 4, y - std / 4, alpha=0.5, color=c)
            axins.fill_between(steps, y + std / 4, y - std / 4, alpha=0.5,
                               color=c)

    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    offset = global_dict['offset_list'][global_dict['i']][global_dict['j']]
    line = global_dict['line_loc'][global_dict['j']]
    if offset is not None:
        mark_inset(ax, axins, loc1=line[0], loc2=line[1], fc=(1, 1, 1), ec="0.5")

    if eval_type == 'offline':
        ax.axhline(y=1, color=PlotColor('optimal'), linestyle='-',
                   label='optimal policy expected reward')

        ax.set_ylim([global_dict['ylim1'][global_dict['j']], global_dict['ylim2'][global_dict['j']]])
        # handles, labels = plt.gca().get_legend_handles_labels()


# def PlotOffline(ax, sim_outputs, req_param, general_sim_params, optimal_policy_reward):
#     for method, parameter, varied_param in sim_outputs.keys():
#         if NeedToPlot(req_param, parameter, method, temporal_extension_run, varied_param):
#
#             y, std, steps = CalcData(general_sim_params, sim_outputs[(method, parameter, varied_param)],
#                                      temp_ext, 'offline',
#                                      optimal_policy_reward)
#
#             if not temporal_extension_run:
#                 c = PlotColor(method, parameter)
#                 ax.plot(steps, y, color=c, label=method + ' ' + str(parameter))
#             else:
#                 c = PlotColor(method, parameter, temp_ext)
#                 if method == 'random':
#                     rand = 'random, '
#                 else:
#                     rand = ''
#                 label = rand + r'$\lambda$ = ' + str(temp_ext)
#                 ax.plot(steps, y, color=c, label=label)
#
#             ax.fill_between(steps, y + std / 4, y - std / 4, alpha=0.1, color=c)
#
#     ax.axhline(y=1, color=PlotColor('optimal'), linestyle='-',
#                label='optimal policy expected reward')
#
#     ax.set_ylim([global_dict['ylim1'][global_dict['j']], global_dict['ylim2'][global_dict['j']]])


def createLabel(varied_param_str, varied_param_val, method, parameter):
    if varied_param_str == 'temporal_extension':
        if method == 'random':
            rand = 'random, '
        else:
            rand = ''
        label = rand + r'$\lambda$ = ' + str(varied_param_val)
        return label
    if varied_param_str == 'agents':
        agent_frac = str(int(varied_param_val[0] / 10)) + '/' + str(int(varied_param_val[1] / 10))
        return agent_frac
        # return method + ' ' + str(parameter) + ' ' + agent_frac + 'agents ratio run'
    return method + ' ' + str(parameter)


def PlotEvaluationForParam(sim_outputs, optimal_policy_reward, req_param, general_sim_params):
    SetDefaults()

    ax = plt.Subplot(global_dict['global_fig'], global_dict['inner'][global_dict['j']])
    global_dict['global_fig'].add_subplot(ax)
    global_dict['axes'][global_dict['i'], global_dict['j']] = ax
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 3))
    if global_dict['i'] == 0:
        PlotData(ax, sim_outputs, req_param, general_sim_params, optimal_policy_reward, 'online')
    if global_dict['i'] == 1:
        PlotData(ax, sim_outputs, req_param, general_sim_params, optimal_policy_reward, 'offline')


def PlotResults(result_list, opt_policy_reward_list, general_sim_params):
    for i, ((mdp_type, res_data), opt_reward) in enumerate(zip(result_list, opt_policy_reward_list)):
        PlotEvaluation(res_data, opt_reward, general_sim_params)
        # if mdp_type in ['chains', 'bridge']:
        # CompareActivations(res_data, i)

        # plt.show()


MAIN_FOLDER = r'C:\Users\Naama\Dropbox\project\report graphs\\'


def ListOfMDPFromPckl():
    # part
    ylim1 = [0.9, 0.7, 0.7, 0.2, 0.7]
    ylim2 = [1.01, 1.03, 1.03, 1.08, 1.03]
    # x1, x2, y1, y2
    offset_list = [[(8300, 8500, 8500, 10000),
                    (7200, 7500, 28000, 32000),
                    (8000, 9000, 20000, 30000),
                    (8000, 9000, 300, 320),
                    (8000, 9000, 20000, 30000)],
                   [None, None, None, None, None, None]]
    zoom_list = [10, 7, 2, 3, 3]
    loc_list = [8, 4, 4, 4, 4]
    line_loc = [(4, 1), (1, 3), (2, 1), (1, 2), (3, 1)]
    titles = ['Cliques', 'Tunnel', 'Tree', 'Cliff']
    graph_name = [r'cliques\TD and Reward\5 actions\run_res2.pckl',
                  # r'star\3 actions\run_res2.pckl',
                  r'tunnel\run_res2_withTD.pckl',
                  r'run_res2.pckl',
                  r'clif\run_res2.pckl'

                  ]
    data_path = [DATA_PATH(name) for name in graph_name]
    mdp_num = len(graph_name)

    res_tuple_list = {'res': [], 'opt_reward': []}
    res_tuple = None
    for i, path in enumerate(data_path):
        res_tuple = pickle.load(open(path, 'rb'))
        res_tuple_list['res'].append(res_tuple['res'][0])
        res_tuple_list['opt_reward'].append(res_tuple['opt_reward'][0])
    res_tuple_list['params'] = res_tuple['params']

    return res_tuple_list, titles, zoom_list, loc_list, ylim1, ylim2, offset_list, line_loc, mdp_num


def ListOfMDPFromPath():
    ylim1 = [0, 0, 0.7, -0.01, 0.65]
    # ylim1 = [0.85, 0.6, 0.7, -0.01, 0.65]
    # ylim1 = [0.95, 0.87, 0.5, 0.95, 0.7]
    ylim2 = [1.01, 1.03, 1.01, 1.05, 1.1]
    # ylim2 = [1.01, 1.03, 1.01, 1.05, 1.1]
    # ylim2 = [1.001, 1.03, 1.005, 1.03, 1.03]
    # titles = ['Tree', 'cliff', 'Clique', 'Tunnel']
    titles = ['Cliques', 'Tunnel', 'Tree', 'Cliff']

    model_free = pickle.load(open(r'C:\Users\Naama\Dropbox\project\model free\4mdps_new.pckl', 'rb'))
    model_free_tunnel = pickle.load(open(r'C:\Users\Naama\Dropbox\project\pnina\run_res2_tunnel.pckl', 'rb'))
    model_free_clique = pickle.load(open(r'C:\Users\Naama\Dropbox\project\model free\model_free_clique.pckl', 'rb'))
    res_tuple_list = pickle.load(open(r'C:\Users\Naama\Dropbox\project\graphs\value function\run_res2.pckl', 'rb'))
    cliff = pickle.load(open(r'C:\Users\Naama\Dropbox\project\graphs\value function\cliff_with_vf.pckl', 'rb'))
    tree = pickle.load(open(r'C:\Users\Naama\Dropbox\project\graphs\value function\tree_with_vf.pckl', 'rb'))

    # model free clique
    model_free['res'][1] = model_free_clique['res'][0]
    # model free tunnel
    model_free['res'][3] = model_free_tunnel['res'][0]
    model_free_index = [0, 2, 1, 3]
    model_free['res'] = [model_free['res'][i] for i in model_free_index]
    model_free['opt_reward'] = [model_free['opt_reward'][i] for i in model_free_index]


    # exchange cliff and tree
    res_tuple_list['res'][1] = cliff['res'][0]
    res_tuple_list['res'][0] = tree['res'][0]
    # res_tuple_list['opt_reward'][1] = cliff['opt_reward'][0]
    # res_tuple_list['opt_reward'][0] = tree['opt_reward'][0]

    # add model free
    for res1, res2 in zip(res_tuple_list['res'], model_free['res']):
        res1['critics'] = {**res1['critics'], **res2['critics']}
        res1['indices'] = {**res1['indices'], **res2['indices']}



    index = [2, 3, 0, 1]
    res_tuple_list['res'] = [res_tuple_list['res'][i] for i in index]
    res_tuple_list['opt_reward'] = [res_tuple_list['opt_reward'][i] for i in index]

    # x1, x2, y1, y2
    offset_list = [[(8300, 8500, 7500, 11000),
                    None,
                    (8000, 9000, 13500, 16800),
                    (7000, 8500, 140, 270),
                    (8000, 9000, 20000, 30000),
                    (8000, 9000, 2000, 3000)],
                   [None, None, None, None, None, None]]
    zoom_list = [10, 3, 2.5, 1.5, 2, 3]
    loc_list = [8, 4, 4, 4, 4, 4]
    line_loc = [(4, 1), (1, 2), (1, 2), (1, 3), (2, 1), (3, 1)]
    # mdp_num = len(res_tuple_list['res'])
    mdp_num = 1
    return model_free_tunnel, titles, zoom_list, loc_list, ylim1, ylim2, offset_list, line_loc, mdp_num


def DATA_PATH(path):
    return MAIN_FOLDER + path


def FormatPlot(mdp_num, varied_param):
    if mdp_num > 1:
        # axes[1, 2].set_xlabel('simulation steps')

        global_dict['axes'][0, 0].set_ylabel('normalized evaluated regret')
        global_dict['axes'][1, 0].set_ylabel('normalized average reward')

        for outer_ax in global_dict['axes']:
            for i, ax in enumerate(outer_ax):
                ax.set_title(global_dict['titles'][i])
            # axes[0, 2].set_title('Regret' + '\n\n' + titles[2])
            # axes[1, 2].set_title('Evaluation')
    else:
        global_dict['axes'][1, 0].set_xlabel('simulation steps')
        global_dict['axes'][0, 0].set_ylabel('evaluated regret')
        global_dict['axes'][1, 0].set_ylabel('average reward')
        # axes[0, 0].set_title('Regret')
        # axes[1, 0].set_title('Evaluation')

    # global_fig.text(.5, .06, 'simulation steps', ha='center', fontweight="bold")
    BuildLegend(mdp_num, varied_param)
    # global_dict['global_fig'].legend()
    # global_fig.show()


def GTRes():
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
    res_tuple = None
    for i, path in enumerate(data_path):
        res_tuple = pickle.load(open(path, 'rb'))
        res_tuple_list['res'].append(res_tuple['res'][0])
        res_tuple_list['opt_reward'].append(res_tuple['opt_reward'][0])
    res_tuple_list['params'] = res_tuple['params']

    ylim1 = [0.7, 0.6, 0.5, 0.3, 0.5, 0.5]
    ylim2 = [1.02, 1.03, 1.05, 1.05, 1.05, 1.05]
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
    titles = ['Cliques', 'Tunnel', 'Tree', 'Cliff']
    return res_tuple_list, titles, zoom_list, loc_list, ylim1, ylim2, offset_list, line_loc, mdp_num


def TERes():
    '''
    res_tuple_add = [pickle.load(open(DATA_PATH(r'temporal_extension\run_res_TE_8.pckl'), 'rb')),
                     pickle.load(open(DATA_PATH(r'temporal_extension\run_res_TE_16.pckl'), 'rb'))]
    res_tuple_list = pickle.load(open(DATA_PATH(r'temporal_extension\run_res.pckl'), 'rb'))

    keys = []
    values = []
    for res_tuple in res_tuple_add:
            [keys.append(key) for key in res_tuple['res'][0][1].keys()]
            [values.append(value) for value in res_tuple['res'][0][1].values()]

    for key, value in zip(keys, values):
        res_tuple_list['res'][0][1][key] = value

    del res_tuple_list['res'][0][1][('gittins', 'reward', 2)]

    ylim1 = [0.7]
    ylim2 = [1.05]
    titles = ['']
    # x1, x2, y1, y2
    offset_list = [[None],[None]]
    zoom_list = [10]
    loc_list = [8]
    line_loc = [(4, 1)]
    mdp_num = len(res_tuple_list['res'])
    '''
    # part
    ylim1 = [0.7, 0.6, 0.5, 0.3, 0.5, 0.5]
    ylim2 = [1.02, 1.03, 1.05, 1.05, 1.05, 1.05]
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
    titles = ['Cliques', 'Tunnel', 'Tree', 'Cliff']
    graph_name = [r'temporal_extension\with random\TE_clique_with_random.pckl',
                  r'temporal_extension\with random\TE_tunnel_with_random.pckl',
                  r'temporal_extension\with random\TE_tunnel_with_random.pckl',
                  r'temporal_extension\with random\TE_cliff_with_random.pckl'
                  ]
    data_path = [DATA_PATH(name) for name in graph_name]
    mdp_num = len(graph_name)

    res_tuple_list = {'res': [], 'opt_reward': []}
    res_tuple = None
    for i, path in enumerate(data_path):
        res_tuple = pickle.load(open(path, 'rb'))
        res_tuple_list['res'].append(res_tuple['res'][0])
        res_tuple_list['opt_reward'].append(res_tuple['opt_reward'][0])
    res_tuple_list['params'] = res_tuple['params']

    return res_tuple_list, titles, zoom_list, loc_list, ylim1, ylim2, offset_list, line_loc, mdp_num


def AgentsRes():
    ylim1 = [0.75, 0.75]
    ylim2 = [1.01, 1.01]
    # x1, x2, y1, y2

    # # k const
    # offset_list = [[None, None, None, None, None, None],
    #                [(5000, 6000, 0.95, 1.005), None, None, None, None, None]]
    # zoom_list = [2, 3, 7, 2, 3, 3]
    # loc_list = [8, 4, 4, 4, 4, 4]
    # line_loc = [(2, 1), (1, 2), (1, 3), (2, 1), (1, 2), (3, 1)]
    # graph_path_list = [r'agents\run_res2_2.pckl']

    # N const
    offset_list = [[None, None, None, None, None, None],
                   [(3500, 4500, 0.95, 1.005), None, None, None, None, None]]
    zoom_list = [2, 3, 7, 2, 3, 3]
    loc_list = [8, 4, 4, 4, 4, 4]
    line_loc = [(2, 1), (1, 2), (1, 3), (2, 1), (1, 2), (3, 1)]
    graph_path_list = [r'agents\run_res2_3.pckl']

    titles = ['Cliques', 'Cliques']

    graph_name = [DATA_PATH(path) for path in graph_path_list]
    mdp_num = len(graph_name)

    res_tuple_list = {'res': [], 'opt_reward': [], 'agents_ratio': []}
    for i, path in enumerate(graph_name):
        res_tuple = pickle.load(open(path, 'rb'))
        res_tuple_list['res'].append(res_tuple['res'][0])
        res_tuple_list['opt_reward'].append(res_tuple['opt_reward'][0])
    res_tuple_list['params'] = res_tuple['params']

    # res_tuple_list = [pickle.load(open(name, 'rb')) for name in graph_name]

    return res_tuple_list, titles, zoom_list, loc_list, ylim1, ylim2, offset_list, line_loc, mdp_num


def AddBadStatesForGT(res_list, general_sim_params):
    ax = plt.Subplot(global_dict['global_fig'], global_dict['inner'][1])
    steps = []
    global_dict['global_fig'].add_subplot(ax)
    # axes[0, 1] = ax
    bad_states = []
    for i, mdp_res in enumerate(res_list):
        bad_states.append(
            np.asarray(mdp_res[1][('gittins', 'reward', 1)]['bad_states'][0]) / general_sim_params['eval_freq'])
        eval_count = int(general_sim_params['steps'] /
                         (general_sim_params['eval_freq']))
        max_step = eval_count * general_sim_params['eval_freq']
        steps = np.linspace(0, max_step, num=eval_count)[:-1]
    # del[bad_states[0]]
    bad_states = np.asarray(bad_states).T
    ax.plot(steps, bad_states)
    ax.set_xlabel('simulation steps')
    ax.set_ylabel('number of wrong decisions per step')
    ax.set_title('Gittins Calculation in Evaluated Model')
    ax.legend(['Tree', 'Clique', 'Cliff', 'Star', 'Tunnel'])
    plt.legend(['Clique', 'Cliff', 'Star', 'Tunnel'])
    # plt.show()


def SetGlobalsFromMain(Results):
    res, titles = Results

    mdp_num = len(titles)

    ylim1 = [0.3]
    ylim2 = [1.1]
    # x1, x2, y1, y2
    offset_list = [[None, None, None, None, None, None],
                   [None, None, None, None, None, None]]
    zoom_list = [10, 3, 7, 2, 3, 3]
    loc_list = [8, 4, 4, 4, 4, 4]
    line_loc = [(4, 1), (1, 2), (1, 3), (2, 1), (1, 2), (3, 1)]

    return res, titles, zoom_list, loc_list, ylim1, ylim2, offset_list, line_loc, mdp_num


def BuildGlobalDict(param):
    res_tuple_list, titles, zoom_list, loc_list, ylim1, ylim2, offset_list, line_loc, mdp_num = param
    global_dict['res_tuple_list'] = res_tuple_list
    global_dict['titles'] = titles
    global_dict['zoom_list'] = zoom_list
    global_dict['loc_list'] = loc_list
    global_dict['ylim1'] = ylim1
    global_dict['ylim2'] = ylim2
    global_dict['offset_list'] = offset_list
    global_dict['line_loc'] = line_loc
    global_dict['mdp_num'] = mdp_num


def SetGlobals(plot_type, Results):
    if plot_type == 'GT':
        res = GTRes()
    elif plot_type == 'TE':
        res = TERes()
    elif plot_type == 'agents':
        res = AgentsRes()
    elif plot_type == 'pickle list':
        res = ListOfMDPFromPckl()
    elif plot_type == 'combined pickle':
        res = ListOfMDPFromPath()
    else:
        res = SetGlobalsFromMain(Results)

    BuildGlobalDict(res)


def PlotResultsWrraper(plot_type, Results=None):
    if plot_type == 'GT':
        PlotGT(Results)
        return
    SetGlobals(plot_type, Results)

    global_dict['global_fig'] = plt.figure(figsize=(6, 8))
    outer = gridspec.GridSpec(2, 1, wspace=0.3, hspace=0.3)

    global_dict['axes'] = np.empty(shape=(2, global_dict['mdp_num']), dtype=object)
    for i in [0, 1]:
        global_dict['i'] = i
        global_dict['inner'] = gridspec.GridSpecFromSubplotSpec(1, global_dict['mdp_num'],
                                                                subplot_spec=outer[i], wspace=0.3, hspace=0.3)

        res_tuple_list = global_dict['res_tuple_list']
        for j in range(global_dict['mdp_num']):
            global_dict['j'] = j
            PlotEvaluation(res_tuple_list['res'][j]['critics'],
                           res_tuple_list['opt_reward'][j],
                           res_tuple_list['params'])
    FormatPlot(global_dict['mdp_num'], res_tuple_list['params']['varied_param'])
    global_dict['global_fig'].subplots_adjust(left=0.15)
    global_dict['global_fig'].show()
    plt.show()


if __name__ == '__main__':
    PlotResultsWrraper('combined pickle')
