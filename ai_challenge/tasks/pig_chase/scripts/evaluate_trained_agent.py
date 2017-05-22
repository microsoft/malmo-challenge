"""
Script that should be used to evaluate the agent.
"""
import os
import argparse

from malmopy.visualization.visualizer import CsvVisualizer

from ai_challenge.experiments import load_wrap_vb_learner
from ai_challenge.tasks.pig_chase.agents import PigChaseChallengeAgent, PigChaseHumanAgent
from ai_challenge.tasks.pig_chase.environment import ENV_AGENT_NAMES, \
    ENV_ACTIONS, CustomStateBuilder, PigChaseSymbolicStateBuilder
from ai_challenge.tasks.pig_chase.scripts.evaluation import PigChaseEvaluator
from ai_challenge.utils import get_results_path, parse_clients_args
from ai_challenge.config import Config

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('clients',
                            nargs='*',
                            default=['127.0.0.1:10000', '127.0.0.1:10001'],
                            help='Minecraft clients endpoints in the form ip:port')
    arg_parser.add_argument('--cfg', '-c',
                            required=True,
                            help='Name of the config from config directory')
    arg_parser.add_argument('--reps', '-r',
                            help='Number of episodes.',
                            type=int,
                            default=1000)

    args = arg_parser.parse_args()
    config = Config(args.cfg)
    clients = parse_clients_args(args.clients)
    path = config.get_str('BASIC', 'load_path')
    saved_dir_nm, saved_learner_nm = os.path.split(path)
    agent = load_wrap_vb_learner(saved_dir_nm,
                                 saved_learner_nm,
                                 args.cfg,
                                 internal_to_store=[],
                                 name=ENV_AGENT_NAMES[1],
                                 nb_actions=len(ENV_ACTIONS))
    agent_state_builder = CustomStateBuilder()
    evaluation = PigChaseEvaluator(clients, agent, agent, agent_state_builder, args.reps)
    evaluation.run()
    evaluation.save('agent_vs_challenge', 'submission_file.json')
