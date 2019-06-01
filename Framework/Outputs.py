from functools import reduce
from itertools import product
from Simulator.Simulator import SimInputFactory, SimulatorFactory
from MDPModel import MDPModel as Mdp, MDPConfig as mdpcfg
import numpy as np
import pickle
from Framework.Plotting import plot_results_wrraper
from Framework import config as cfg


class Runner:
    """
    Auxiliary class which holds executes all the requested runs, and outputs their result.
    Initiation requires all simulation parameters, which are shared between all runs.
    Executing also requires an iterable of all MDPs.
    """
    sim_params = None
    varying = None
    gt_compare = None

    def __init__(self, sim_params: cfg.SimulationParameters, load):
        Runner.varying = sim_params.varied_definition_str
        Runner.sim_params = sim_params
        Runner.gt_compare = sim_params.gt_compare

        self.res = None
        self.res_address = sim_params.results_address

        def create_definitions(param):
            return list(product([param], sim_params.method_dict[param], getattr(sim_params, Runner.varying)))

        self.definitions = reduce(lambda a, b: a + b, map(create_definitions, sim_params.method_dict.keys()))

        def gen_mdps(load_mdps):
            def generate_mdp_list(type_list):
                """
                Generate new MDP per type in type_list.
                Note that MDPs are generated according to default config. insert values to constructors to override them.
                """

                def generate_mdp(mdp_type):
                    if mdp_type == 'tunnel': return Mdp.ChainsTunnelMDP(mdpcfg.TunnelMDPConfig())
                    if mdp_type == 'star': return Mdp.StarMDP(mdpcfg.StarMDPConfig())
                    if mdp_type == 'clique': return Mdp.CliquesMDP(mdpcfg.CliqueMDPConfig())
                    if mdp_type == 'cliff': return Mdp.CliffWalker(mdpcfg.CliffMDPConfig())
                    if mdp_type == 'directed': return Mdp.DirectedTreeMDP(mdpcfg.DirectedTreeMDPConfig())

                    raise NotImplementedError()

                mdps = [generate_mdp(mdp_type) for mdp_type in type_list]

                with open('mdp.pckl', 'wb') as f1:
                    pickle.dump(mdps, f1)

                return mdps

            def load_mdp_list():
                """load previously generated MDPs"""
                # clique = pickle.load(open("mdp.pckl", "rb"))
                # directed = pickle.load(open("directed_mdp_with_gittins.pckl", "rb"))
                # clique = pickle.load(open("clique_mdp_with_gittins.pckl", "rb"))
                # cliff = pickle.load(open("cliff_mdp_with_gittins.pckl", "rb"))
                # star = pickle.load(open("star_mdp_with_gittins.pckl", "rb"))
                tunnel = pickle.load(open("tunnel_mdp_with_gittins.pckl", "rb"))

                mdps = [tunnel[0]]
                # mdps = [directed[0], clique[0], cliff[0], tunnel[0]]

                return mdps

            return load_mdp_list() if load_mdps else generate_mdp_list(sim_params.mdp_types)

        self.mdp_list = gen_mdps(load)

    def run(self):
        """Executes a run for each mdp stored"""

        def run_mdp(mdp_idx, mdp):
            """Executes required runs for a single MDP. Creates and returns an updated MDPResult objects"""
            print(f'run MDP num {mdp_idx}')
            return MDPResult(mdp).run(self.definitions)

        self.res = list(map(lambda e: run_mdp(*e), enumerate(self.mdp_list)))

        with open(self.res_address, 'wb') as f:
            pickle.dump(self.res, f)

        return self.res

    def plot(self, titles):
        if Runner.gt_compare:
            plot_results_wrraper('GT', (self.res, titles))
        else:
            plot_results_wrraper('from main', (self.res, titles))


class MDPResult:
    """Executes required runs for a single MDP, and holds results, alongside it's optimal reward"""

    def __init__(self, mdp):
        self.mdp: mdp.MDPModel = mdp
        self.optimal_reward = self.mdp.calc_opt_expected_reward()
        self.result = None

    def run(self, definitions):
        def run_mdp_with_defs(method, parameter, varied_definition):
            """Run every configuration for the required amount of times. Returns results"""
            print(f'    running {method}, prioritizing using {parameter}, with {Runner.varying} = '
                  f'{varied_definition}:')
            setattr(Runner.sim_params, Runner.varying, varied_definition)
            res = ResFactory.generate(Runner.gt_compare, self.mdp)

            for idx in range(Runner.sim_params.runs_per_mdp):
                print(f'     start run #{idx + 1}')
                simulator = SimulatorFactory(self.mdp, Runner.sim_params, Runner.gt_compare)
                res.update(simulator.simulate((SimInputFactory(method, parameter, Runner.sim_params))))

            res.summarize_critics()
            return res

        self.result = {definition: run_mdp_with_defs(*definition) for definition in definitions}
        return self


class RunResult:
    """Container for the results of all runs of a single configuration"""

    def __init__(self, mdp):
        self.mdp_type = mdp.type
        self.critics = []

    def update(self, critics):
        """Add new result to the container"""
        self.critics.append(critics)

    def summarize_critics(self):
        """After all runs of the configuration were made, summarize results, in preparations for presentation"""
        def critic_getter_generator(attr_name):
            return list(map(lambda x: x.value_vec[attr_name], self.critics))
        res = {
            'online': (np.cumsum(np.mean(critic_getter_generator('online')), axis=0),
                       np.std(critic_getter_generator('online'), axis=0)),
            'offline': (np.mean(critic_getter_generator('offline'), axis=0),
                        np.std(critic_getter_generator('offline'), axis=0)),
            'bad_states': list(map(lambda critic: np.diff(critic.bad_activated_states), self.critics)),
            'critics': self.critics}

        if self.mdp_type in ['chains', 'bridge']:
            res['chain_activations'] = np.mean(list(map(lambda critic: critic.chain_activations, self.critics)), axis=0)

        return res


class GTRunResult(RunResult):
    """In Ground Truth runs we also track calculated Gittins indices versus true indices"""

    def __init__(self, mdp):
        super().__init__(mdp)
        self.indices = []
        self.gt = []

    def update(self, critics, **kwargs):
        """Add new result to the container"""
        super().update(critics)
        self.indices.append(kwargs['indices'])
        self.gt.append(kwargs['gt'])


class ResFactory:
    @staticmethod
    def generate(gt_compare, mdp):
        return GTRunResult(mdp) if gt_compare else RunResult(mdp)
