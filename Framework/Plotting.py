import numpy as np
import matplotlib.pyplot as plt
from Simulator.Simulator import *
from Framework.Inputs import ChainSimulationOutput


def CompareActivations(data_output, mdp_i):
    plt.figure()
    tick_shift = np.linspace(-0.35, 0.35, len(data_output))
    # tick_shift = [-0.25, -0.05, 0.15, 0.35]
    chain_num = len(data_output[0][0].chain_activation)
    [plt.bar([tick_shift[_iter] + s for s in range(chain_num)], data_output[_iter][0].chain_activation, width=0.1,
             align='center')
     for _iter in range(len(data_output))]

    plt.xticks(range(chain_num), ['chain ' + str(s) for s in range(chain_num)])
    plt.legend([data_output[_iter][1] for _iter in range(len(data_output))])
    plt.title('Agents Activation per Chains for mdp num ' + str(mdp_i))


def PlotEvaluation(data_output, optimal_policy_reward, general_sim_params):
    params = ['reward', 'error', 'all']
    [PlotEvaluationForParam(data_output, optimal_policy_reward, param, general_sim_params)
     for param in params]


def PlotEvaluationForParam(data_output, optimal_policy_reward, param, general_sim_params):
    fig, ax = plt.subplots(nrows=1, ncols=len(general_sim_params['eval_type']))
    eval_count = int(np.ceil(general_sim_params['steps'] / general_sim_params['eval_freq']))
    steps = np.array(list(range(eval_count))) * general_sim_params['eval_freq']
    for _iter in range(len(data_output)):
        if data_output[_iter][2] == param or param == 'all':
            for i, eval_type in enumerate(general_sim_params['eval_type']):
                reward_eval = data_output[_iter][0].reward_eval.get(eval_type)
                smoothed_eval = np.array([smooth(reward_eval[i])[:-10] for i in range(reward_eval.shape[0])])

                y = np.mean(smoothed_eval, axis=0)
                std = np.std(smoothed_eval, axis=0)
                ax[i].plot(steps, y, label=data_output[_iter][1])
                ax[i].fill_between(steps, y + std / 2, y - std / 2, alpha=0.5)

    for i, eval_type in enumerate(general_sim_params['eval_type']):
        ax[i].set_title(eval_type)
        ax[i].set_xlabel('simulation steps')
        ax[i].set_ylabel('evaluated reward')
        if eval_type == 'offline':
            plt.axhline(y=optimal_policy_reward, color='r', linestyle='-', label='optimal policy expected reward')
        ax[i].legend()
    # elif eval_type == 'online':
    #     plt.plot(steps, optimal_policy_reward, 'optimal policy expected reward')
    # method_type.insert(0, 'optimal policy expected reward')

    title = 'Reward Evaluation'
    title += (' - agents prioritized by ' + param) if param != 'all' else ''
    fig.suptitle(title + '\naverage of ' + str(general_sim_params['runs_per_mdp']) + ' runs')


def PlotResults(results, _opt_policy_reward, general_sim_params):
    for mdp_i in range(len(results)):
        res = results[mdp_i]

        data = reduce(lambda a, b: a + b, [[(res[method][param], str(method) + ' ' + str(param), param)
                                            for param in res[method].keys()] for method in res.keys()])
        PlotEvaluation(data, _opt_policy_reward[mdp_i], general_sim_params)
        try:
            CompareActivations(data, mdp_i)
        except TypeError:
            pass

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


