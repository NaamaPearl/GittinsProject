import pickle
import matplotlib.pyplot as plt
import numpy as np
from Framework.Plotting import smooth


def PlotLookAhead(result_list, param):
    steps = 5000
    eval_freq = 50
    eval_count = int(np.ceil(steps / eval_freq))
    steps = np.array(list(range(eval_count))) * eval_freq
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.suptitle(param + ' Based Gittins Prioritization \nlook ahead comparison')

    for i, eval_type in enumerate(['online', 'offline']):
        for idx, result in enumerate(result_list):
            reward_eval = result.reward_eval.get(eval_type)
            smoothed_eval = np.array([smooth(reward_eval[i])[:-10] for i in range(reward_eval.shape[0])])
            y = np.mean(smoothed_eval, axis=0)
            std = np.std(smoothed_eval, axis=0)
            ax[i].plot(steps, y, label=r'$\lambda$ = ' + str(idx))
            ax[i].fill_between(steps, y + std / 2, y - std / 2, alpha=0.5)

        ax[i].set_xlabel('simulation steps')
        ax[i].set_ylabel('evaluated reward')
        ax[i].set_title(eval_type)
        ax[i].legend()

    ax[1].axhline(y=809, color='r', linestyle='-', label='optimal policy expected reward')
    plt.show()


if __name__ == '__main__':
    with open('temporal_results1.pickle', 'rb') as handle:
        temporal_results_list = pickle.load(handle)

    reward_result = []
    error_result = []
    for temporal_result in temporal_results_list:
        reward_result.append(temporal_result[0]['gittins']['reward'])
        error_result.append(temporal_result[0]['gittins']['error'])

    PlotLookAhead(reward_result, 'Reward')
    PlotLookAhead(error_result, 'Error')

    print('all done')
