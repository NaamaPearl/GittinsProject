import matplotlib.pyplot as plt
from framework import *
from ModelSimulator import *


def CompareActivations(vectors, chain_num, method_type):
    plt.figure()
    tick_shift = [-0.25, -0.05, 0.15, 0.35]
    [plt.bar([tick_shift[i] + s for s in range(chain_num)], vectors[method_type[i]], width=0.1, align='center')
     for i in range(len(vectors))]

    plt.xticks(range(chain_num), ['chain ' + str(s) for s in range(4)])
    plt.legend(method_type)
    plt.title('Agents Activation per Chains')


def PlotEvaluation(vectors, method_type, optimal_policy_reward):
    plt.figure()
    [plt.plot(vectors[method_type[i]]) for i in range(len(vectors))]
    plt.axhline(y=optimal_policy_reward, color='r', linestyle='-')
    plt.legend(method_type)
    plt.title('Reward Eval')


def RunSimulationsOnMdp(mdp, simulation_steps, agents_to_run, runs_for_specific_mdp, method_type_list, gamma, trajectory_len):
    # creating simulation
    simulators = {method: SimulatorFactory(method, mdp, agents_to_run, gamma) for method in method_type_list}
    simulator_inputs = {method: SimInputFactory(method, simulation_steps, agents_to_run, trajectory_len) for method in method_type_list}

    chain_activation = {key: 0 for key in method_type_list}
    reward_eval = {key: 0 for key in method_type_list}

    for i in range(runs_for_specific_mdp):

        for method in method_type_list:
            simulators[method].simulate(simulator_inputs[method])
            chain_activation[method] += (
                    np.asarray(simulators[method].critic.chain_activations) / runs_for_specific_mdp)
            reward_eval[method] += (np.asarray(simulators[method].evaluate_policy) / runs_for_specific_mdp)
            # print('simulate finished, %s agents activated' % sum(simulators[method].critic.chain_activations))

    return chain_activation, reward_eval


if __name__ == '__main__':
    n = 21
    method_type_list = ['sweeping']
    # method_type_list = ['random', 'error', 'reward', 'sweeping']
    mdp_num = 1
    gamma = 0.9
    trajectory_len = 50

    for i in range(mdp_num):
        mdp = SeperateChainsMDP(n=n, reward_param=((0, 0, 0), (5, 1, 1)), reward_type='gauss', gamma=gamma, trajectory_len=trajectory_len)

        activations, reward_eval = RunSimulationsOnMdp(mdp,
                                                       simulation_steps=5000,
                                                       agents_to_run=10,
                                                       runs_for_specific_mdp=1,
                                                       method_type_list=method_type_list,
                                                       gamma=gamma,
                                                       trajectory_len=trajectory_len)
        CompareActivations(activations, 2, method_type_list)
        PlotEvaluation(reward_eval, method_type_list, mdp.CalcOptExpectedReward(trajectory_len))

    print('all done')
