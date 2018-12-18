import matplotlib.pyplot as plt
from ModelSimulator import *
from framework import *


def CompareActivations(data_output):
    plt.figure()
    tick_shift = np.linspace(-0.35, 0.35, len(data_output))
    # tick_shift = [-0.25, -0.05, 0.15, 0.35]
    chain_num = len(data_output[0][0].chain_activation)
    [plt.bar([tick_shift[_iter] + s for s in range(chain_num)], data_output[_iter][0].chain_activation, width=0.1, align='center')
     for _iter in range(len(data_output))]

    plt.xticks(range(chain_num), ['chain ' + str(s) for s in range(4)])
    plt.legend([data_output[_iter][1] for _iter in range(len(data_output))])
    plt.title('Agents Activation per Chains')


def PlotEvaluation(data_output, optimal_policy_reward):
    plt.figure()

    for _iter in range(len(data_output)):
        reward_eval = np.array(data_output[_iter][0].reward_eval)
        std = np.std(reward_eval, axis=0)
        # plt.plot(reward_eval)
        plt.errorbar(x=list(range(len(std))), y=np.mean(reward_eval, axis=0), yerr=std, marker='^')
    plt.axhline(y=optimal_policy_reward, color='r', linestyle='-')
    method_type = [data_output[_iter][1] for _iter in range(len(data_output))]
    method_type.insert(0, 'optimal policy value')
    plt.legend(method_type)
    plt.title('Reward Eval')


def RunSimulationsOnMdp(simulators, simulation_inputs, runs_per_mdp):
    simulation_outputs = {method: {parameter: ChainSimulationOutput() for parameter in method_dict[method]}
                          for method in method_dict.keys()}

    for i in range(runs_per_mdp):
        print('     run number ' + str(i))
        for method in simulators.keys():
            print('method: ' + method)
            simulator = simulators[method]
            for simulation_input in simulation_inputs[method]:
                simulator.simulate(simulation_input)

                simulation_output = simulation_outputs[method][simulation_input.parameter]
                simulation_output.chain_activation += (
                        np.asarray(simulator.critic.chain_activations) / runs_per_mdp)
                simulation_output.reward_eval.append(np.asarray(simulator.critic.value_vec))
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
        print('run MDP num ' + str(i))
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
    n = 31
    mdp_num = 1 # TODO doesnt work yet for more than one
    gamma = 0.9
    trajectory_len = 50
    eval_type = 'online'
    agents_to_run = 10
    # method_dict = {'random': [None], 'greedy': ['reward', 'error']}
    method_dict = {'random': [None], 'gittins': ['reward', 'error'], 'greedy': ['reward', 'error']}
    mdp_list = [SeperateChainsMDP(n=n,
                                  reward_param={1: {'bernoulli_p': 1, 'gauss_params': (5, 1, 1)},
                                                2: {'bernoulli_p': 0.1, 'gauss_params': (50, 1, 1)}},
                                  reward_type='gauss',
                                  gamma=gamma,
                                  chain_num=3)
                for _ in range(mdp_num)]
    simulation_steps = 5000

    opt_policy_reward = [mdp.CalcOptExpectedReward(trajectory_len) for mdp in mdp_list]
    res = RunSimulations(method_dict, mdp_list, runs_per_mdp=5)
    PlotResults(res, opt_policy_reward)
    print('all done')
