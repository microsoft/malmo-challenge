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
from os import path
from threading import Thread, active_count
from time import sleep

from malmopy.agent import RandomAgent, TemporalMemory, LinearEpsilonGreedyExplorer
from malmopy.model.chainer import QNeuralNetwork, DQNChain

from common import parse_clients_args, visualize_training, ENV_AGENT_NAMES
from agent import PigChaseChallengeAgent, PigChaseQLearnerAgent
from environment import PigChaseEnvironment, PigChaseSymbolicStateBuilder
from malmopy.environment.malmo import MalmoALEStateBuilder

from malmopy.agent import QLearnerAgent, TemporalMemory

try:
    from malmopy.visualization.tensorboard import TensorboardVisualizer
    from malmopy.visualization.tensorboard.cntk import CntkConverter
except ImportError:
    print('Cannot import tensorboard, using ConsoleVisualizer.')
    from malmopy.visualization import ConsoleVisualizer

# Enforce path
sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.join(os.path.pardir, os.getcwd()))

DQN_FOLDER = 'results/baselines/%s/dqn/%s-%s'
EPOCH_SIZE = 250000


def agent_factory(name, role, clients, device, max_epochs, logdir, visualizer):

    assert len(clients) >= 2, 'Not enough clients (need at least 2)'
    clients = parse_clients_args(clients)

    if role == 0:

        builder = PigChaseSymbolicStateBuilder()
        env = PigChaseEnvironment(clients, builder, role=role,
                                  randomize_positions=True)
        agent = PigChaseChallengeAgent(name)

        obs = env.reset()
        reward = 0
        agent_done = False

        while True:
            if env.done:
                obs = env.reset()

            # select an action
            action = agent.act(obs, reward, agent_done, is_training=True)
            # take a step
            obs, reward, agent_done = env.do(action)

    else:
        env = PigChaseEnvironment(clients, MalmoALEStateBuilder(),
                                  role=role, randomize_positions=True)
        memory = TemporalMemory(100000, (84, 84))
        chain = DQNChain((memory.history_length, 84, 84), env.available_actions)
        target_chain = DQNChain((memory.history_length, 84, 84), env.available_actions)
        model = QNeuralNetwork(chain, target_chain, device)
        explorer = LinearEpsilonGreedyExplorer(1, 0.1, 1000000)
        agent = PigChaseQLearnerAgent(name, env.available_actions,
                                      model, memory, 0.99, 32, 50000,
                                      explorer=explorer, visualizer=visualizer)

        obs = env.reset()
        reward = 0
        agent_done = False
        viz_rewards = []

        max_training_steps = EPOCH_SIZE * max_epochs
        for step in range(1, max_training_steps+1):

            # check if env needs reset
            if env.done:

                visualize_training(visualizer, step, viz_rewards)
                agent.inject_summaries(step)
                viz_rewards = []
                obs = env.reset()

            # select an action
            action = agent.act(obs, reward, agent_done, is_training=True)
            # take a step
            obs, reward, agent_done = env.do(action)
            viz_rewards.append(reward)

            if (step % EPOCH_SIZE) == 0:
                if 'model' in locals():
                    model.save('pig_chase-dqn_%d.model' % (step / EPOCH_SIZE))


def run_experiment(agents_def):
    assert len(agents_def) == 2, 'Not enough agents (required: 2, got: %d)' \
                                 % len(agents_def)

    processes = []
    for agent in agents_def:
        p = Thread(target=agent_factory, kwargs=agent)
        p.daemon = True
        p.start()

        # Give the server time to start
        if agent['role'] == 0:
            sleep(1)

        processes.append(p)

    try:
        # wait until only the challenge agent is left
        while active_count() > 2:
            sleep(0.1)
    except KeyboardInterrupt:
        print('Caught control-c - shutting down.')


if __name__ == '__main__':
    arg_parser = ArgumentParser('Pig Chase DQN experiment')
    arg_parser.add_argument('-e', '--epochs', type=int, default=5,
                            help='Number of epochs to run.')
    arg_parser.add_argument('clients', nargs='*',
                            default=['127.0.0.1:10000', '127.0.0.1:10001'],
                            help='Minecraft clients endpoints (ip(:port)?)+')
    arg_parser.add_argument('-d', '--device', type=int, default=-1,
                            help='GPU device on which to run the experiment.')
    args = arg_parser.parse_args()

    logdir = path.join('results/pig_chase/dqn', datetime.utcnow().isoformat())
    if 'malmopy.visualization.tensorboard' in sys.modules:
        visualizer = TensorboardVisualizer()
        visualizer.initialize(logdir, None)

    else:
        visualizer = ConsoleVisualizer()

    agents = [{'name': agent, 'role': role, 'clients': args.clients,
               'device': args.device, 'max_epochs': args.epochs,
               'logdir': logdir, 'visualizer': visualizer}
              for role, agent in enumerate(ENV_AGENT_NAMES)]

    run_experiment(agents)
