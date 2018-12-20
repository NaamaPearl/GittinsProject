import matplotlib.pyplot as plt
from ModelSimulator import *
from framework import *


def CompareActivations(data_output):
    plt.figure()
    tick_shift = np.linspace(-0.35, 0.35, len(data_output))
    # tick_shift = [-0.25, -0.05, 0.15, 0.35]
    chain_num = len(data_output[0][0].chain_activation)
    [plt.bar([tick_shift[_iter] + s for s in range(chain_num)], data_output[_iter][0].chain_activation, width=0.1,
             align='center')
     for _iter in range(len(data_output))]

    plt.xticks(range(chain_num), ['chain ' + str(s) for s in range(4)])
    plt.legend([data_output[_iter][1] for _iter in range(len(data_output))])
    plt.title('Agents Activation per Chains')


def PlotEvaluation(data_output, optimal_policy_reward):
    params = ['all', 'reward', 'error']
    [PlotEvaluationForParam(data_output, optimal_policy_reward, param) for param in params]

def PlotEvaluationForParam(data_output, optimal_policy_reward, param):
    plt.figure()
    method_type = []
    for _iter in range(len(data_output)):
        if data_output[_iter][2] == param:
            reward_eval = np.array(data_output[_iter][0].reward_eval)
            std = np.std(reward_eval, axis=0)
            plt.errorbar(x=list(range(len(std))), y=np.mean(reward_eval, axis=0), yerr=std, marker='^')
            method_type.append(data_output[_iter][1])
    if len(opt_policy_reward) == 1:
        plt.axhline(y=optimal_policy_reward, color='r', linestyle='-')
    else:
        plt.plot(optimal_policy_reward[0])
    method_type.insert(0, 'optimal policy value')
    plt.legend(method_type)
    plt.title('Reward Eval - prioritize by ' + param)


def RunSimulationsOnMdp(simulators, simulation_inputs, runs_per_mdp, sim_params):
    simulation_outputs = {
        method: {parameter: ChainSimulationOutput() for parameter in sim_params['method_dict'][method]}
        for method in sim_params['method_dict'].keys()}

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


def RunSimulations(_mdp_list, runs_per_mdp, _sim_params):
    simulators = [{
        method: SimulatorFactory(method, mdp, _sim_params)
        for method in _sim_params['method_dict'].keys()} for mdp in _mdp_list]

    simulation_inputs = {method: [
        SimInputFactory(method, parameter, _sim_params)
        for parameter in _sim_params['method_dict'][method]] for method in _sim_params['method_dict'].keys()}

    result = []
    for i in range(len(mdp_list)):
        print('run MDP num ' + str(i))
        result.append(RunSimulationsOnMdp(simulators[i], simulation_inputs, runs_per_mdp, _sim_params))
    return simulators, result


def PlotResults(results, opt_policy_reward):
    for i in range(len(results)):
        res = results[i]

        data = reduce(lambda a, b: a + b, [[(res[method][param], str(method) + ' ' + str(param), param)
                                            for param in res[method].keys()] for method in res.keys()])
        CompareActivations(data)
        PlotEvaluation(data, opt_policy_reward[i])


if __name__ == '__main__':
    # building the MDP's
    n = 31
    mdp_num = 1  # TODO doesnt work yet for more than one
    gamma = 0.9
    mdp_list = [SeperateChainsMDP(n=n,
                                  reward_param={1: {'bernoulli_p': 1, 'gauss_params': (10, 1, 3)},
                                                'trap': {'bernoulli_p': 1, 'gauss_params': (-10, 0, 0)},
                                                'leads_to_trap': {'bernoulli_p': 1, 'gauss_params': (4, 0, 0)}},
                                  gamma=gamma,
                                  traps_num=0,
                                  chain_num=2)
                for _ in range(mdp_num)]

    # define general simulation params
    general_sim_params = {'method_dict': {'gittins': ['reward', 'error'], 'greedy': ['reward', 'error']},
                          'steps': 1000,
                          'eval_type': 'offline',
                          'agents_to_run': 10,
                          'trajectory_len': 50,
                          'gamma': gamma,
                          'eval_freq': 50}

    opt_policy_reward = [mdp.CalcOptExpectedReward(general_sim_params) for mdp in mdp_list]
    simulators, res = RunSimulations(mdp_list, runs_per_mdp=1, _sim_params=general_sim_params)
    PlotResults(res, opt_policy_reward)

    print('all done')
