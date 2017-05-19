import os
import sys
from argparse import ArgumentParser
from datetime import datetime
from multiprocessing import Process, Event
from os import path
from time import sleep

from malmopy.agent.gui import ARROW_KEYS_MAPPING
from malmopy.visualization import ConsoleVisualizer

from ai_challenge.utils import parse_clients_args
from ai_challenge.tasks.pig_chase.agents import PigChaseHumanAgent
from ai_challenge.tasks.pig_chase.environment import PigChaseEnvironment, \
    PigChaseSymbolicStateBuilder, ENV_AGENT_NAMES, ENV_ACTIONS, CustomStateBuilder
from ai_challenge.experiments import load_wrap_vb_learner
from ai_challenge.visualization import PlotDataManager, Plotter

# Enforce path
sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.join(os.path.pardir, os.getcwd()))


MAX_ACTIONS = 25  # this should match the mission definition, used for display only


def agent_factory(name, role, kind, clients, max_episodes, max_actions, logdir, quit):
    assert len(clients) >= 2, 'There are not enough Malmo clients in the pool (need at least 2)'

    clients = parse_clients_args(clients)
    visualizer = ConsoleVisualizer(prefix='Agent %d' % role)

    if role == 0:
        env = PigChaseEnvironment(clients, CustomStateBuilder(),
                                  actions=ENV_ACTIONS, role=role,
                                  human_speed=True, randomize_positions=True)

        saved_dir_nm = 'rec_dqn_exp/2017-05-07T15:48:57.564765'
        saved_learner_nm = '36014'
        model_used = 'DQN'
        agent = load_wrap_vb_learner(saved_dir_nm, saved_learner_nm,
                                     model_used=model_used,
                                     internal_to_store=[],
                                     name=ENV_AGENT_NAMES[1],
                                     nb_actions=len(ENV_ACTIONS))
        hidden_states_dict = agent.get_model_states_ref(['h2', 'rec_h1', 'lstm_c'])

        plot_data_manager = PlotDataManager(prop_dict=hidden_states_dict, step_dict={})
        plotter = Plotter(plot_data_manager)
        plotter.start()

        obs = env.reset()
        reward = 0
        rewards = []
        done = False
        episode = 0

        while True:

            # select an action
            action = agent.act(obs, reward, done, True)

            if done:
                rewards = []
                episode += 1
                obs = env.reset()

            # take a step
            obs, reward, done = env.do(action)
            rewards.append(reward)

    else:
        env = PigChaseEnvironment(clients, PigChaseSymbolicStateBuilder(),
                                  actions=list(ARROW_KEYS_MAPPING.values()),
                                  role=role, randomize_positions=True)
        env.reset(PigChaseEnvironment.AGENT_TYPE_3)

        agent = PigChaseHumanAgent(name, env, list(ARROW_KEYS_MAPPING.keys()),
                                   max_episodes, max_actions, visualizer, quit)
        agent.show()


def run_mission(agents_def):
    assert len(agents_def) == 2, 'Incompatible number of agents (required: 2, got: %d)' % len(
        agents_def)
    quit = Event()
    processes = []
    for agent in agents_def:
        agent['quit'] = quit
        p = Process(target=agent_factory, kwargs=agent)
        p.daemon = True
        p.start()

        if agent['role'] == 0:
            sleep(1)  # Just to let time for the server to start

        processes.append(p)
    quit.wait()
    for process in processes:
        process.terminate()


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-e', '--episodes', type=int, default=10,
                            help='Number of episodes to run.')
    arg_parser.add_argument('-k', '--kind', type=str, default='astar', choices=['astar', 'random'],
                            help='The kind of agent to play with (random or astar).')
    arg_parser.add_argument('clients', nargs='*',
                            default=['127.0.0.1:10000', '127.0.0.1:10001'],
                            help='Malmo clients (ip(:port)?)+')
    args = arg_parser.parse_args()

    logdir = path.join('results/pig-human', datetime.utcnow().isoformat())
    agents = [{'name': agent, 'role': role, 'kind': args.kind,
               'clients': args.clients, 'max_episodes': args.episodes,
               'max_actions': MAX_ACTIONS, 'logdir': logdir}
              for role, agent in enumerate(ENV_AGENT_NAMES)]

    run_mission(agents)
