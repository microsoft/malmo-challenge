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
from time import sleep
from json import dump
from os.path import exists, join, pardir, abspath
from os import makedirs
from numpy import mean, var

from malmopy.visualization.visualizer import CsvVisualizer
from ai_challenge.tasks.pig_chase.agents import PigChaseChallengeAgent
from ai_challenge.tasks.pig_chase.environment import PigChaseEnvironment, ENV_AGENT_NAMES, \
    PigChaseSymbolicStateBuilder

# Enforce path
sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.join(os.path.pardir, os.getcwd()))

EVAL_EPISODES = 1000.


class PigChaseEvaluator(object):
    def __init__(self, clients, agent_100k, state_builder, out_dir):
        assert len(clients) >= 2, 'Not enough clients provided'

        self._clients = clients
        self._agent_100k = agent_100k
        self._state_builder = state_builder
        self.out_dir = out_dir
        self._accumulators = {'100k': []}

    def save(self, experiment_name, filepath):
        """
        Save the evaluation results in a JSON file 
        understandable by the leaderboard.
        
        Note: The leaderboard will not accept a submission if you already 
        uploaded a file with the same experiment name.
        
        :param experiment_name: An identifier for the experiment
        :param filepath: Path where to store the results file
        :return: 
        """

        assert experiment_name is not None, 'experiment_name cannot be None'

        # Compute metrics
        metrics = {key: {'mean': mean(buffer),
                         'var': var(buffer),
                         'count': len(buffer)}
                   for key, buffer in self._accumulators.items()}

        metrics['experimentname'] = experiment_name

        filepath = abspath(filepath)
        parent = join(pardir, filepath)
        if not exists(parent):
            makedirs(parent)

        with open(filepath, 'w') as f_out:
            dump(metrics, f_out)

        print('==================================')
        print('Evaluation done, results written at %s' % filepath)


    def run(self):
        from multiprocessing import Process

        env = PigChaseEnvironment(self._clients, self._state_builder,
                                  role=1, randomize_positions=True)
        print('==================================')
        print('Starting evaluation of Agent @100k')

        p = Process(target=run_challenge_agent, args=(self._clients, self.out_dir))
        p.start()
        sleep(5)
        agent_loop(self._agent_100k, env, self._accumulators['100k'])
        p.terminate()
        print('stats:', {key: {'mean': mean(buffer),
                               'var': var(buffer),
                               'count': len(buffer)}
                         for key, buffer in self._accumulators.items()})


def run_challenge_agent(clients, out_dir):
    builder = PigChaseSymbolicStateBuilder()
    agent = PigChaseChallengeAgent(ENV_AGENT_NAMES[0], visualizer=CsvVisualizer(
        output_file=os.path.join(out_dir, 'challenge_agent_type.csv')))
    env = PigChaseEnvironment(clients, builder, role=0,
                              randomize_positions=True)
    agent_loop(agent, env, None)
    agent._visualizer.close()


def agent_loop(agent, env, metrics_acc):
    agent_done = False
    reward = 0
    episode = 0
    obs = env.reset()

    while episode < EVAL_EPISODES:
        # check if env needs reset
        if env.done:
            print('Episode %d (%.2f)%%' % (episode, (episode / EVAL_EPISODES) * 100.))
            obs = env.reset()
            while obs is None:
                # this can happen if the episode ended with the first
                # action of the other agent
                print('Warning: received obs == None.')
                obs = env.reset()

            episode += 1

        # select an action
        action = agent.act(obs, reward, agent_done, is_training=True)
        # take a step
        obs, reward, agent_done = env.do(action)

        if metrics_acc is not None:
            metrics_acc.append(reward)
