from Framework.Plotting import *


def RunSimulations(mdp_list, _sim_params):
    simulators = [{
        method: {parameter: SimulatorFactory(mdp, _sim_params)
                 for parameter in _sim_params['method_dict'][method]} for method in _sim_params['method_dict'].keys()}
        for mdp in mdp_list]

    simulation_inputs = {method: [
        SimInputFactory(method, parameter, _sim_params)
        for parameter in _sim_params['method_dict'][method]] for method in _sim_params['method_dict'].keys()}

    result = []
    for i in range(len(mdp_list)):
        print('run MDP num ' + str(i))
        result.append(RunSimulationsOnMdp(simulators[i], simulation_inputs, _sim_params))
    return simulators, result


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


if __name__ == '__main__':
    # building the MDPs
    tunnel_length = 5
    _mdp_list = [ChainsTunnelMDP(n=46, actions=4, succ_num=2, op_succ_num=4, chain_num=3, gamma=0.99, traps_num=0,
                                 tunnel_indexes=list(range(37, 37 + tunnel_length)),
                                 reward_param={2: {'bernoulli_p': 1, 'gauss_params': ((10, 1), 0)},
                                               'lead_to_tunnel': {'bernoulli_p': 1, 'gauss_params': ((-1, 0), 0)},
                                               'tunnel_end': {'bernoulli_p': 1, 'gauss_params': ((100, 0), 0)}})]

    # mdp_list = [StarMDP(n=31, actions=3, succ_num=5, op_succ_num=10, chain_num=3, gamma=0.9,
    #                     reward_param={0: {'bernoulli_p': 1, 'gauss_params': ((0, 1), 1)},
    #                                   1: {'bernoulli_p': 1, 'gauss_params': ((0, 1), 1)},
    #                                   2: {'bernoulli_p': 1, 'gauss_params': ((0, 1), 1)}
    #                                   })]

    # define general simulation params
    _method_dict = {'gittins': ['reward', 'error'], 'greedy': ['reward', 'error'], 'random': [None]}
    general_sim_params = {'method_dict': _method_dict,
                          'steps': 5000, 'eval_type': ['online', 'offline'], 'agents_to_run': 10, 'agents_ratio': 6,
                          'trajectory_len': 100, 'eval_freq': 50, 'epsilon': 0.1, 'reset_freq': 8000, 'grades_freq': 50,
                          'gittins_discount': 1, 'gittins_look_ahead': 1, 'T_bored': 3,
                          'runs_per_mdp': 3}

    opt_policy_reward = [mdp.CalcOptExpectedReward(general_sim_params) for mdp in _mdp_list]
    _simulators, res = RunSimulations(_mdp_list, _sim_params=general_sim_params)
    PlotResults(res, opt_policy_reward, general_sim_params)

    print('all done')
