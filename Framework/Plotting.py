import matplotlib.pyplot as plt
from Simulator.Simulator import *
import pickle
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset



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
            return 'xkcd:magenta'
        if method == 'greedy':
            return 'xkcd:pink'

    ## Reward
    if param == 'reward':
        if method == 'gittins':
            return 'xkcd:bright blue'
        if method == 'greedy':
            return 'xkcd:sky blue'


def PlotEvaluationForParam(sim_outputs, optimal_policy_reward, req_param, general_sim_params):
    fig, ax = plt.subplots(nrows=1, ncols=len(general_sim_params['eval_type']))
    # sub region of the original image
    axins = zoomed_inset_axes(ax[0], 3.5, loc=4)
    x1, x2, y1, y2 = 7800, 8200, 12000, 19000
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    temporal_extension_run = len(general_sim_params['temporal_extension']) > 1

    for method, parameter, temp_ext in sim_outputs.keys():
        if parameter == req_param or req_param == 'all':
            eval_count = int(general_sim_params['steps'] /
                             (general_sim_params['eval_freq'] * temp_ext))
            max_step = eval_count * general_sim_params['eval_freq'] * temp_ext
            samples = np.linspace(0, max_step, num=eval_count)
            steps = np.linspace(0, max_step, num=eval_count * temp_ext)

            for i, eval_type in enumerate(general_sim_params['eval_type']):
                mean_values, std_tmp = sim_outputs[(method, parameter, temp_ext)].get(eval_type)
                mean_values_smooth = np.array(smooth(mean_values))
                y = np.interp(steps, samples, mean_values_smooth)
                std = np.interp(steps, samples, std_tmp)

                if not temporal_extension_run:
                    ax[i].plot(steps, y, color=PlotColor(method, parameter), label=method + ' ' + str(parameter))
                    if i == 0:axins.plot(steps, y, color=PlotColor(method, parameter), label=method + ' ' + str(parameter))
                else:
                    ax[i].plot(steps, y, color=PlotColor(method, parameter), label=r'$\lambda$ = ' + str(temp_ext))
                    if i == 0:axins.plot(steps, y, color=PlotColor(method, parameter), label=r'$\lambda$ = ' + str(temp_ext))
                ax[i].fill_between(steps, y + std / 4, y - std / 4, alpha=0.5, color=PlotColor(method, parameter))
                if i == 0: axins.fill_between(steps, y + std / 4, y - std / 4, alpha=0.5, color=PlotColor(method, parameter))

    ## set just evaluation with log y scale
    ax[1].set_yscale("log", nonposy='clip')



    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax[0], axins, loc1=2, loc2=4, fc="none", ec="0.5")

    for i, eval_type in enumerate(general_sim_params['eval_type']):
        ax[i].set_title('Evaluation' if eval_type == 'offline' else 'Regret')
        ax[i].set_xlabel('simulation steps')
        if eval_type == 'offline':
            ax[i].set_ylabel('average reward')
            ax[1].axhline(y=optimal_policy_reward, color=PlotColor('optimal'), linestyle='-', label='optimal policy expected reward')
        else:
            ax[i].set_ylabel('evaluated regret')

        handles, labels = ax[i].get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0][::-1]))
        labels = list(labels)
        handles = list(handles)
        rand_idx = labels.index('random None') if 'random None' in labels else None
        if rand_idx is not None:
            labels.remove('random None')
            labels.append('random')
            handles.append(handles.pop(rand_idx))
        ax[i].legend(handles, labels)

    title = 'Reward Evaluation' if not temporal_extension_run else 'Temporal Extension Comparision'
    title += (' - agents prioritized by ' + req_param) if req_param != 'all' else ''
    fig.suptitle(title + '\naverage of ' + str(general_sim_params['runs_per_mdp']) + ' runs')


def PlotResults(result_list, opt_policy_reward_list, general_sim_params):
    for i, ((mdp_type, res_data), opt_reward) in enumerate(zip(result_list, opt_policy_reward_list)):
        PlotEvaluation(res_data, opt_reward, general_sim_params)
        if mdp_type in ['chains', 'bridge']:
            CompareActivations(res_data, i)

        plt.show()


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    addition = int((window_len - 1) / 2)
    return y[addition:-addition]


if __name__ == '__main__':
    res_tuple = pickle.load(open('..\\run_res2.pckl', 'rb'))

    PlotResults(res_tuple['res'], res_tuple['opt_reward'], res_tuple['params'])
