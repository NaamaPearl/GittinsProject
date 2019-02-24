import matplotlib.pyplot as plt
from Simulator.Simulator import *
import pickle


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
    params = ['reward', 'error', 'all']
    [PlotEvaluationForParam(data_output, optimal_policy_reward, param, general_sim_params)
     for param in params]


def PlotEvaluationForParam(sim_outputs, optimal_policy_reward, param, general_sim_params):
    fig, ax = plt.subplots(nrows=1, ncols=len(general_sim_params['eval_type']))
    eval_count = int(general_sim_params['steps'] /
                             (general_sim_params['eval_freq'] * general_sim_params['temporal_extension']))
    steps = np.array(list(range(eval_count))) * general_sim_params['eval_freq']

    for definitions in sim_outputs.keys():
        if definitions[1] == param or param == 'all':
            for i, eval_type in enumerate(general_sim_params['eval_type']):
                mean_values, std = sim_outputs[definitions].get(eval_type)
                # y = np.array(smooth(mean_values)[:-10])
                y = mean_values

                ax[i].plot(steps, y, label=definitions[0] + ' ' + str(definitions[1]))
                ax[i].fill_between(steps, y + std / 4, y - std / 4, alpha=0.5)

    for i, eval_type in enumerate(general_sim_params['eval_type']):
        ax[i].set_title(eval_type if eval_type == 'offline' else 'Regret')
        ax[i].set_xlabel('simulation steps')
        if eval_type == 'offline':
            ax[i].set_ylabel('evaluated reward')
            plt.axhline(y=optimal_policy_reward, color='r', linestyle='-', label='optimal policy expected reward')
        else:
            ax[i].set_ylabel('evaluated regret')
        ax[i].legend()

    title = 'Reward Evaluation'
    title += (' - agents prioritized by ' + param) if param != 'all' else ''
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
    return y


if __name__ == '__main__':
    res_tuple = pickle.load(open('..\\run_res2.pckl', 'rb'))

    PlotResults(res_tuple['res'], res_tuple['opt_reward'], res_tuple['params'])



