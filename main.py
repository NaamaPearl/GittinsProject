from Framework import config as cfg
import argparse
from Framework.Outputs import Runner


if __name__ == '__main__':
    ''''Parse user's arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_mdps', '-l', action='store_true', help='load existing MDP')
    args = parser.parse_args()

    '''define general simulation params. Add arguments to constructor, or use default values'''
    general_sim_params = cfg.SimulationParameters()

    runner = Runner(general_sim_params, args.load_mdps)
    runner.run()
    runner.plot(['tree'])
