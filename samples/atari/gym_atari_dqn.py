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

from argparse import ArgumentParser
from datetime import datetime
from subprocess import Popen

from malmopy.agent import QLearnerAgent, TemporalMemory
from malmopy.environment.gym import GymEnvironment

try:
    from malmopy.visualization.tensorboard import TensorboardVisualizer
    from malmopy.visualization.tensorboard.cntk import CntkConverter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    print('Cannot import tensorboard, using ConsoleVisualizer.')
    from malmopy.visualization import ConsoleVisualizer

    TENSORBOARD_AVAILABLE = False


ROOT_FOLDER = 'results/baselines/%s/dqn/%s-%s'
EPOCH_SIZE = 250000


def visualize_training(visualizer, step, rewards, tag='Training'):
    visualizer.add_entry(step, '%s/reward per episode' % tag, sum(rewards))
    visualizer.add_entry(step, '%s/max.reward' % tag, max(rewards))
    visualizer.add_entry(step, '%s/min.reward' % tag, min(rewards))
    visualizer.add_entry(step, '%s/actions per episode' % tag, len(rewards)-1)


def run_experiment(environment, backend, device_id, max_epoch, record, logdir,
                   visualizer):

    env = GymEnvironment(environment,
                         monitoring_path=logdir if record else None)

    if backend == 'cntk':
        from malmopy.model.cntk import QNeuralNetwork as CntkDQN
        model = CntkDQN((4, 84, 84), env.available_actions, momentum=0.95,
                        device_id=device_id, visualizer=visualizer)
    else:
        from malmopy.model.chainer import DQNChain, QNeuralNetwork as ChainerDQN
        chain = DQNChain((4, 84, 84), env.available_actions)
        target_chain = DQNChain((4, 84, 84), env.available_actions)
        model = ChainerDQN(chain, target_chain,
                           momentum=0.95, device_id=device_id)

    memory = TemporalMemory(1000000, model.input_shape[1:])
    agent = QLearnerAgent("DQN Agent", env.available_actions, model, memory,
                          0.99, 32, train_after=50000, reward_clipping=(-1, 1),
                          visualizer=visualizer)

    state = env.reset()
    reward = 0
    agent_done = False
    viz_rewards = []

    max_training_steps = max_epoch * EPOCH_SIZE
    for step in range(1, max_training_steps + 1):

        # check if env needs reset
        if env.done:
            visualize_training(visualizer, step, viz_rewards)
            agent.inject_summaries(step)
            viz_rewards = []
            state = env.reset()

        # select an action
        action = agent.act(state, reward, agent_done, is_training=True)

        # take a step
        state, reward, agent_done = env.do(action)
        viz_rewards.append(reward)

        if (step % EPOCH_SIZE) == 0:
            model.save('%s-%s-dqn_%d.model' %
                       (backend, environment, step / EPOCH_SIZE))


if __name__ == '__main__':
    arg_parser = ArgumentParser(description='OpenAI Gym DQN example')
    arg_parser.add_argument('-b', '--backend', type=str, default='cntk',
                            choices=['cntk', 'chainer'],
                            help='Neural network backend to use.')
    arg_parser.add_argument('-d', '--device', type=int, default=-1,
                            help='GPU device on which to run the experiment.')
    arg_parser.add_argument('-r', '--record', action='store_true',
                            help='Setting this will record runs')
    arg_parser.add_argument('-e', '--epochs', type=int, default=50,
                            help='Number of epochs. One epoch is 250k actions.')
    arg_parser.add_argument('-p', '--port', type=int, default=6006,
                            help='Port for running tensorboard.')
    arg_parser.add_argument('env', type=str, metavar='environment',
                            nargs='?', default='Breakout-v3',
                            help='Gym environment to run')

    args = arg_parser.parse_args()

    logdir = ROOT_FOLDER % (args.env, args.backend, datetime.utcnow().isoformat())
    if TENSORBOARD_AVAILABLE:
        visualizer = TensorboardVisualizer()
        visualizer.initialize(logdir, None)
        print('Starting tensorboard ...')
        p = Popen(['tensorboard', '--logdir=results', '--port=%d' % args.port])

    else:
        visualizer = ConsoleVisualizer()

    print('Starting experiment')
    run_experiment(args.env, args.backend, int(args.device), args.epochs,
                   args.record, logdir, visualizer)

