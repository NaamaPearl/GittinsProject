import matplotlib.pyplot as plt
from ModelSimulator import *
from framework import *


def CompareActivations(data_output):
    plt.figure()
    tick_shift = [-0.25, -0.05, 0.15, 0.35]
    chain_num = len(data_output[0][0].chain_activation)
    [plt.bar([tick_shift[_iter] + s for s in range(chain_num)], data_output[_iter][0].chain_activation, width=0.1, align='center')
     for _iter in range(len(data_output))]

    plt.xticks(range(chain_num), ['chain ' + str(s) for s in range(4)])
    plt.legend([data_output[_iter][1] for _iter in range(len(data_output))])
    plt.title('Agents Activation per Chains')


def PlotEvaluation(data_output, optimal_policy_reward):
    plt.figure()
    [plt.plot(data_output[_iter][0].reward_eval) for _iter in range(len(data_output))]
    plt.axhline(y=optimal_policy_reward, color='r', linestyle='-')
    method_type = [data_output[_iter][1] for _iter in range(len(data_output))]
    method_type.append('optimal policy value')
    plt.legend(method_type)
    plt.title('Reward Eval')


def RunSimulationsOnMdp(simulators, simulation_inputs, runs_per_mdp):
    simulation_outputs = {method: {parameter: ChainSimulationOutput() for parameter in method_dict[method]}
                          for method in method_dict.keys()}

    for i in range(runs_per_mdp):
        for method in simulators.keys():
            simulator = simulators[method]
            for simulation_input in simulation_inputs[method]:
                simulator.simulate(simulation_input)

                simulation_output = simulation_outputs[method][simulation_input.parameter]
                simulation_output.chain_activation += (
                        np.asarray(simulator.critic.chain_activations) / runs_per_mdp)
                simulation_output.reward_eval += (np.asarray(simulator.critic.value_vec) / runs_per_mdp)
            # print('simulate finished, %s agents activated' % sum(simulators[method].critic.chain_activations))

    return simulation_outputs


def RunSimulations(_method_dict, _mdp_list, runs_per_mdp):
    simulators = [{
        method: SimulatorFactory(method, mdp, agents_to_run, gamma, eval_type)
        for method in _method_dict.keys()} for mdp in _mdp_list]

    simulation_inputs = {method: [
        SimInputFactory(method, parameter, simulation_steps, agents_to_run, trajectory_len)
        for parameter in method_dict[method]] for method in method_dict.keys()}

    result = []
    for i in range(len(mdp_list)):
        result.append(RunSimulationsOnMdp(simulators[i], simulation_inputs, runs_per_mdp))

    return result


def PlotResults(results, opt_policy_reward):
    for i in range(len(results)):
        res = results[i]

        data = reduce(lambda a, b: a + b, [[(res[method][param], str(method) + ' ' + str(param))
                                           for param in res[method].keys()] for method in res.keys()])
        CompareActivations(data)
        PlotEvaluation(data, opt_policy_reward)


if __name__ == '__main__':
    n = 21
    mdp_num = 1 # TODO doesnt work yet for more than one
    gamma = 0.9
    trajectory_len = 50
    eval_type = 'online'
    agents_to_run = 10
    method_dict = {'random': [None], 'greedy': ['reward', 'error']}
    # method_dict = {'random': [None], 'gittins': ['reward', 'error'], 'greedy': ['reward', 'error']}
    mdp_list = [SeperateChainsMDP(n=n, reward_param=((0, 0, 0), (5, 1, 1)), reward_type='gauss', gamma=gamma,
                                  trajectory_len=trajectory_len) for _ in range(mdp_num)]
    simulation_steps = 1000

    opt_policy_reward = [mdp.CalcOptExpectedReward(trajectory_len) for mdp in mdp_list]
    PlotResults(RunSimulations(method_dict, mdp_list, runs_per_mdp=1), opt_policy_reward)

    print('all done')
