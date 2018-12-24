import matplotlib.pyplot as plt
import pickle
from Simulator.Simulator import *


def CompareActivations(data_output, mdp_i):
    plt.figure()
    tick_shift = np.linspace(-0.35, 0.35, len(data_output))
    # tick_shift = [-0.25, -0.05, 0.15, 0.35]
    chain_num = len(data_output[0][0].chain_activation)
    [plt.bar([tick_shift[_iter] + s for s in range(chain_num)], data_output[_iter][0].chain_activation, width=0.1,
             align='center')
     for _iter in range(len(data_output))]

    plt.xticks(range(chain_num), ['chain ' + str(s) for s in range(4)])
    plt.legend([data_output[_iter][1] for _iter in range(len(data_output))])
    plt.title('Agents Activation per Chains for mdp num ' + str(mdp_i))


def PlotEvaluation(data_output, optimal_policy_reward, mdp_i, eval_type, eval_freq):
    params = ['reward', 'error', 'all']
    for param in params:
        data_by_param = [data for data in data_output if data[2] == param or param == 'all']
        PlotEvaluationForParam(data_by_param, optimal_policy_reward, mdp_i, eval_type, eval_freq, param)


def PlotEvaluationForParam(data_by_param, optimal_policy_reward, mdp_i, eval_type_list, eval_freq, param):
    fig, ax = plt.subplots(nrows=1, ncols=len(eval_type_list))
    steps = np.array(list(range(len(data_by_param[0][0].reward_eval['online'][0])))) * eval_freq
    for data in data_by_param:
        reward_eval = {eval_type: np.array(data[0].reward_eval[eval_type]) for eval_type in eval_type_list}
        mean = {eval_type: np.mean(reward_eval[eval_type], axis=0) for eval_type in eval_type_list}
        std = {eval_type: np.std(reward_eval[eval_type], axis=0) for eval_type in eval_type_list}

        [ax[i].plot(steps, mean[eval_type], label=data[1]) for i, eval_type in enumerate(eval_type_list)]
        [ax[i].fill_between(steps, mean[eval_type] + std[eval_type]/2, mean[eval_type] - std[eval_type]/2, alpha=0.5)
         for i, eval_type in enumerate(eval_type_list)]

    plt.legend()
    plt.xlabel('simulation steps')
    plt.ylabel('evaluated reward')
    # plt.title(eval_type_list + ' Reward Evaluation - prioritize agents by ' + param
    #           + '\naverage of ' + str(len(data_by_param[0].reward_eval)) + ' runs'
    #           + '\nfor mdp num ' + str(mdp_i))


def RunSimulationsOnMdp(simulators, simulation_inputs, runs_per_mdp, sim_params):
    simulation_outputs = {
        method: {parameter: ChainSimulationOutput(sim_params['eval_type']) for parameter in sim_params['method_dict'][method]}
        for method in sim_params['method_dict'].keys()}

    for i in range(runs_per_mdp):
        print('     run number ' + str(i))
        for method in simulators.keys():
            print('method: ' + method)
            for simulation_input in simulation_inputs[method]:
                simulator = simulators[method][simulation_input.parameter]
                simulator.simulate(simulation_input)

                simulation_output = simulation_outputs[method][simulation_input.parameter]
                simulation_output.chain_activation += (
                        np.asarray(simulator.critic.chain_activations) / runs_per_mdp)
                [simulation_output.reward_eval[eval_type].append(simulator.critic.value_vec[eval_type]) for eval_type in sim_params['eval_type']]
            # print('simulate finished, %s agents activated' % sum(simulators[method].critic.chain_activations))

    return simulation_outputs


def RunSimulations(_mdp_list, runs_per_mdp, _sim_params):
    simulators = [{
        method: {parameter: SimulatorFactory(method, mdp, _sim_params)
                 for parameter in _sim_params['method_dict'][method]} for method in _sim_params['method_dict'].keys()}
        for mdp in _mdp_list]

    simulation_inputs = {method: [
        SimInputFactory(method, parameter, _sim_params)
        for parameter in _sim_params['method_dict'][method]] for method in _sim_params['method_dict'].keys()}

    result = []
    for i in range(len(mdp_list)):
        print('run MDP num ' + str(i))
        result.append(RunSimulationsOnMdp(simulators[i], simulation_inputs, runs_per_mdp, _sim_params))
    return simulators, result


def PlotResults(results, _opt_policy_reward, eval_type, eval_freq):
    for mdp_i in range(len(results)):
        res = results[mdp_i]

        data = reduce(lambda a, b: a + b, [[(res[method][param], str(method) + ' ' + str(param), param)
                                            for param in res[method].keys()] for method in res.keys()])
        CompareActivations(data, mdp_i)
        PlotEvaluation(data, _opt_policy_reward[mdp_i], mdp_i, eval_type, eval_freq)


if __name__ == '__main__':
    # building the MDPs
    mdp_num = 1
    tunnel_length = 5
    load = False
    if load:
        mdp_list = []
        with open('pnina', 'rb') as f:
            mdp_list.append(pickle.load(f))
    else:
        mdp_list = [ChainsTunnelMDP(n=31, action=4, succ_num=2, op_succ_num=5, chain_num=2, gamma=0.9, traps_num=0,
                                    tunnel_indexes=list(range(17, 17 + tunnel_length)),
                                    reward_param={1: {'bernoulli_p': 1, 'gauss_params': ((10, 3), 1)},
                                                  'lead_to_tunnel': {'bernoulli_p': 1, 'gauss_params': ((-1, 0), 0)},
                                                  'tunnel_end': {'bernoulli_p': 1, 'gauss_params': ((100, 0), 0)}})
                    for _ in range(mdp_num)]

    # define general simulation params
    _method_dict = {'random': [None], 'gittins': ['reward', 'error'], 'greedy': ['reward', 'error']}
    # _method_dict = {'greedy': ['reward', 'error']}
    general_sim_params = {'method_dict': _method_dict,
                          'steps': 2000, 'eval_type': ['online', 'offline'], 'agents_to_run': 10, 'trajectory_len': 100,
                          'eval_freq': 50, 'epsilon': 0.1, 'reset_freq': 500000, 'grades_freq': 10,
                          'gittins_look_ahead': tunnel_length, 'gittins_discount': 1, 'T_bored': 1}

    opt_policy_reward = [mdp.CalcOptExpectedReward(general_sim_params) for mdp in mdp_list]
    simulators, res = RunSimulations(mdp_list, runs_per_mdp=2, _sim_params=general_sim_params)
    PlotResults(res, opt_policy_reward, general_sim_params['eval_type'], general_sim_params['eval_freq'])

    print('all done')
