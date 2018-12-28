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
                y = np.mean(reward_eval, axis=0)
                std = np.std(reward_eval, axis=0)
                ax[i].plot(steps, y, label=data_output[_iter][1])
                ax[i].fill_between(steps, y + std / 2, y - std / 2, alpha=0.5)

    for i, eval_type in enumerate(general_sim_params['eval_type']):
        ax[i].set_title(eval_type)
        ax[i].legend()
        ax[i].set_xlabel('simulation steps')
        ax[i].set_ylabel('evaluated reward')
        if eval_type == 'offline':
            plt.axhline(y=optimal_policy_reward, color='r', linestyle='-', label='optimal policy expected reward')
    # elif eval_type == 'online':
    #     plt.plot(steps, optimal_policy_reward, 'optimal policy expected reward')
    # method_type.insert(0, 'optimal policy expected reward')

    title = 'Reward Evaluation'
    title += (' - agents prioritized by ' + param) if param != 'all' else ''
    fig.suptitle(title + '\naverage of ' + str(general_sim_params['runs_per_mdp']) + ' runs')


def RunSimulationsOnMdp(simulators, simulation_inputs, sim_params):
    simulation_outputs = {
        method: {parameter: ChainSimulationOutput(sim_params['eval_type'])
                 for parameter in sim_params['method_dict'][method]}
        for method in sim_params['method_dict'].keys()}

    runs_per_mdp = sim_params['runs_per_mdp']
    for i in range(runs_per_mdp):
        print('     run number ' + str(i))
        for method in simulators.keys():
            print('method: ' + method)
            for simulation_input in simulation_inputs[method]:
                simulator = simulators[method][simulation_input.parameter]
                simulator.simulate(simulation_input)

                simulation_output = simulation_outputs[method][simulation_input.parameter]
                simulation_output.reward_eval.add(simulator.critic.value_vec)
                try:
                    simulation_output.chain_activation += (
                                np.asarray(simulator.critic.chain_activations) / runs_per_mdp)
                except AttributeError:
                    pass

    return simulation_outputs


def RunSimulations(_mdp_list, _sim_params):
    simulators = [{
        method: {parameter: SimulatorFactory(mdp, _sim_params)
                 for parameter in _sim_params['method_dict'][method]} for method in _sim_params['method_dict'].keys()}
        for mdp in _mdp_list]

    simulation_inputs = {method: [
        SimInputFactory(method, parameter, _sim_params)
        for parameter in _sim_params['method_dict'][method]] for method in _sim_params['method_dict'].keys()}

    result = []
    for i in range(len(mdp_list)):
        print('run MDP num ' + str(i))
        result.append(RunSimulationsOnMdp(simulators[i], simulation_inputs, _sim_params))
    return simulators, result


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


if __name__ == '__main__':
    # building the MDPs
    # mdp_num = 1
    tunnel_length = 5
    load = False
    if load:
        mdp_list = []
        with open('pnina', 'rb') as f:
            mdp_list.append(pickle.load(f))
    else:
        # mdp_list = [TreeMDP(n=31, actions=4, succ_num=2, op_succ_num=5, chain_num=2, gamma=0.9, traps_num=0,
        #                     tunnel_indexes=list(range(17, 17 + tunnel_length)), resets_num=3,
        #                     reward_param={1: {'bernoulli_p': 1, 'gauss_params': ((10, 3), 1)},
        #                                   'lead_to_tunnel': {'bernoulli_p': 1, 'gauss_params': ((-1, 0), 0)},
        #                                   'tunnel_end': {'bernoulli_p': 1, 'gauss_params': ((100, 0), 0)}})]

        mdp_list = [StarMDP(n=31, actions=4, succ_num=1, op_succ_num=1, chain_num=5, gamma=0.9,
                            reward_param={'final_state': {'bernoulli_p': 1, 'gauss_params': ((100, 0), 1)},
                                          'line_state': {'bernoulli_p': 1, 'gauss_params': ((0, 1), 1)}})]

    # define general simulation params
    _method_dict = {'gittins': ['reward', 'error'], 'greedy': ['reward', 'error'], 'random': [None]}
    # _method_dict = {'random': [None]}
    general_sim_params = {'method_dict': _method_dict,
                          'steps': 8000, 'eval_type': ['online', 'offline'], 'agents_to_run': 10, 'agents_ratio': 3,
                          'trajectory_len': 100, 'eval_freq': 50, 'epsilon': 0.1, 'reset_freq': 8000, 'grades_freq': 10,
                          'gittins_look_ahead': tunnel_length, 'gittins_discount': 1, 'T_bored': 1,
                          'runs_per_mdp': 2}

    opt_policy_reward = [mdp.CalcOptExpectedReward(general_sim_params) for mdp in mdp_list]
    simulators, res = RunSimulations(mdp_list, _sim_params=general_sim_params)
    PlotResults(res, opt_policy_reward, general_sim_params)

    print('all done')
