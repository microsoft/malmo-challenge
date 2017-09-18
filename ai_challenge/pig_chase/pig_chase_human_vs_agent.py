# Copyright (c) 2017 Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ===================================================================================================================

import os
import sys
from argparse import ArgumentParser
from datetime import datetime
from multiprocessing import Process, Event
from os import path
from time import sleep

from malmopy.agent import RandomAgent
from malmopy.agent.gui import ARROW_KEYS_MAPPING
from malmopy.visualization import ConsoleVisualizer

# Enforce path
sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.join(os.path.pardir, os.getcwd()))

from common import parse_clients_args, ENV_AGENT_NAMES, ENV_ACTIONS, ENV_AGENT_TYPES
from agent import PigChaseChallengeAgent, PigChaseHumanAgent, TabularQLearnerAgent, FocusedAgent, get_agent_type
from environment import PigChaseEnvironment, PigChaseSymbolicStateBuilder

MAX_ACTIONS = 25 # this should match the mission definition, used for display only

def agent_factory(name, role, kind, clients, max_episodes, max_actions, logdir, quit, model_file):
    assert len(clients) >= 2, 'There are not enough Malmo clients in the pool (need at least 2)'

    clients = parse_clients_args(clients)
    visualizer = ConsoleVisualizer(prefix='Agent %d' % role)

    if role == 0:
        env = PigChaseEnvironment(clients, PigChaseSymbolicStateBuilder(),
                                  actions=ENV_ACTIONS, role=role,
                                  human_speed=True, randomize_positions=True)
        if kind == 'challenge':
            agent = PigChaseChallengeAgent(name)
        elif kind == 'astar':
            agent = FocusedAgent(name, ENV_TARGET_NAMES[0])
        elif kind == 'tabq':
            agent = TabularQLearnerAgent(name)
            if model_file != '':
                agent.load(model_file)
        else:
            agent = RandomAgent(name ,env.available_actions)

        obs = env.reset(get_agent_type(agent))
        reward = 0
        rewards = []
        done = False
        episode = 0

        while True:

            # select an action
            action = agent.act(obs, reward, done, True)

            if done:
                visualizer << (episode + 1, 'Reward', sum(rewards))
                rewards = []
                episode += 1

                obs = env.reset(get_agent_type(agent))

            # take a step
            obs, reward, done = env.do(action)
            rewards.append(reward)

    else:
        env = PigChaseEnvironment(clients, PigChaseSymbolicStateBuilder(),
                                  actions=list(ARROW_KEYS_MAPPING.values()),
                                  role=role, randomize_positions=True)
        env.reset(ENV_AGENT_TYPES.HUMAN)

        agent = PigChaseHumanAgent(name, env, list(ARROW_KEYS_MAPPING.keys()),
                                   max_episodes, max_actions, visualizer, quit)
        agent.show()


def run_mission(agents_def):
    assert len(agents_def) == 2, 'Incompatible number of agents (required: 2, got: %d)' % len(agents_def)
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
    arg_parser.add_argument('-e', '--episodes', type=int, default=10, help='Number of episodes to run.')
    arg_parser.add_argument('-k', '--kind', type=str, default='challenge', choices=['astar', 'random', 'tabq', 'challenge'],
                            help='The kind of agent to play with (random, astar, tabq or challenge).')
    arg_parser.add_argument('-m', '--model_file', type=str, default='', help='Model file with which to initialise agent, if appropriate')
    arg_parser.add_argument('clients', nargs='*',
                            default=['127.0.0.1:10000', '127.0.0.1:10001'],
                            help='Malmo clients (ip(:port)?)+')
    args = arg_parser.parse_args()

    logdir = path.join('results/pig-human', datetime.utcnow().isoformat())
    agents = [{'name': agent, 'role': role, 'kind': args.kind, 'model_file': args.model_file,
               'clients': args.clients, 'max_episodes': args.episodes,
               'max_actions': MAX_ACTIONS, 'logdir': logdir}
              for role, agent in enumerate(ENV_AGENT_NAMES)]

    run_mission(agents)
