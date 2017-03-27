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

from __future__ import absolute_import

from collections import namedtuple

import numpy as np

from . import BaseAgent, ReplayMemory, BaseExplorer, LinearEpsilonGreedyExplorer
from ..model import QModel
from ..util import get_rank


class TemporalMemory(ReplayMemory):
    """
    Temporal memory adds a new dimension to store N previous samples (t, t-1, t-2, ..., t-N)
    when sampling from memory
    """

    def __init__(self, max_size, sample_shape, history_length=4,
                 unflicker=False):
        """
        :param max_size: Maximum number of elements in the memory
        :param sample_shape: Shape of each sample
        :param history_length: Length of the visual memory (n previous frames) included with each state
        :param unflicker: Indicate if we need to compute the difference between consecutive frames
        """
        super(TemporalMemory, self).__init__(max_size, sample_shape)

        self._unflicker = unflicker
        self._history_length = max(1, history_length)
        self._last = np.zeros(sample_shape)

    def append(self, state, action, reward, is_terminal):
        if self._unflicker:
            max_diff_buffer = np.maximum(self._last, state)
            self._last = state
            state = max_diff_buffer

        super(TemporalMemory, self).append(state, action, reward, is_terminal)

        if is_terminal:
            if self._unflicker:
                self._last.fill(0)

    def sample(self, size, replace=True):
        """
        Generate a random minibatch. The returned indices can be retrieved using #get_state().
        See the method #minibatch() if you want to retrieve samples directly
        :param size: The minibatch size
        :param replace: Indicate if one index can appear multiple times (True), only once (False)
        :return: Indexes of the sampled states
        """

        if not replace:
            assert (self._count - 1) - self._history_length >= size, \
                'Cannot sample %d from %d elements' % (
                    size, (self._count - 1) - self._history_length)

        # Local variable access are faster in loops
        count, pos, history_len, terminals = self._count - 1, self._pos, \
                                             self._history_length, self._terminals
        indexes = []

        while len(indexes) < size:
            index = np.random.randint(history_len, count)

            # Check if replace=False to not include same index multiple times
            if replace or index not in indexes:

                # if not wrapping over current pointer,
                # then check if there is terminal state wrapped inside
                if not (index >= pos > index - history_len):
                    if not terminals[(index - history_len):index].any():
                        indexes.append(index)

        assert len(indexes) == size
        return indexes

    def get_state(self, index):
        """
        Return the specified state with the visual memory
        :param index: State's index
        :return: Tensor[history_length, input_shape...]
        """
        index %= self._count
        history_length = self._history_length

        # If index > history_length, take from a slice
        if index >= history_length:
            return self._states[(index - (history_length - 1)):index + 1, ...]
        else:
            indexes = np.arange(index - self._history_length + 1, index + 1)
            return self._states.take(indexes, mode='wrap', axis=0)

    @property
    def unflicker(self):
        """
        Indicate if samples added to the replay memory are preprocessed
        by taking the maximum between current frame and previous one
        :return: True if preprocessed, False otherwise
        """
        return self._unflicker

    @property
    def history_length(self):
        """
        Visual memory length 
        (ie. the number of previous frames included for each sample)
        :return: Integer >= 0
        """
        return self._history_length


class History(object):
    """
    Accumulator keeping track of the N previous frames to be used by the agent
    for evaluation
    """

    def __init__(self, shape):
        self._buffer = np.zeros(shape, dtype=np.float32)

    @property
    def value(self):
        return self._buffer

    def append(self, state):
        self._buffer[:-1] = self._buffer[1:]
        self._buffer[-1, ...] = state

    def reset(self):
        self._buffer.fill(0)


# Track previous state and action for observation
Tracker = namedtuple('Tracker', ['state', 'action'])


class QLearnerAgent(BaseAgent):
    def __init__(self, name, nb_actions, model, memory, gamma=.99,
                 minibatch_size=32, train_after=50000, train_frequency=4,
                 explorer=None, reward_clipping=None, visualizer=None):

        assert isinstance(model, QModel), 'model should inherit from QModel'
        assert get_rank(model.input_shape) > 1, 'input_shape rank should be > 1'
        assert isinstance(memory, ReplayMemory), 'memory should inherit from ' \
                                                 'ReplayMemory'
        assert 0 < gamma < 1, 'gamma should be 0 < gamma < 1 (got: %d)' % gamma
        assert minibatch_size > 0, 'minibatch_size should be > 0 (got: %d)' % minibatch_size
        assert train_after >= 0, 'train_after should be >= 0 (got %d)' % train_after
        assert train_frequency > 0, 'train_frequency should be > 0'

        super(QLearnerAgent, self).__init__(name, nb_actions, visualizer)

        self._model = model
        self._memory = memory
        self._gamma = gamma
        self._minibatch_size = minibatch_size
        self._train_after = train_after
        self._train_frequency = train_frequency
        self._history = History(model.input_shape)
        self._actions_taken = 0
        self._tracker = None

        # Rewards clipping related
        reward_clipping = reward_clipping or (-2 ** 31 - 1, 2 ** 31 - 1)
        assert isinstance(reward_clipping, tuple) and len(reward_clipping) == 2, \
            'clip_reward should be None or (min_reward, max_reward)'

        assert reward_clipping[0] <= reward_clipping[1], \
            'max reward_clipping should be >= min (got %d < %d)' % (
                reward_clipping[1], reward_clipping[0])

        self._reward_clipping = reward_clipping

        # Explorer related
        explorer = explorer or LinearEpsilonGreedyExplorer(1, 0.1, 1e6)
        assert isinstance(explorer, BaseExplorer), \
            'explorer should inherit from BaseExplorer'

        self._explorer = explorer

        # Stats related
        self._stats_rewards = []
        self._stats_mean_qvalues = []
        self._stats_stddev_qvalues = []
        self._stats_loss = []

    def act(self, new_state, reward, done, is_training=False):

        if self._tracker is not None:
            self.observe(self._tracker.state, self._tracker.action,
                         reward, new_state, done)

        if is_training:
            if self._actions_taken > self._train_after:
                self.learn()

        # Append the new state to the history
        self._history.append(new_state)

        # select the next action
        if self._explorer.is_exploring(self._actions_taken):
            new_action = self._explorer(self._actions_taken, self.nb_actions)
        else:
            q_values = self._model.evaluate(self._history.value)
            new_action = q_values.argmax()

            self._stats_mean_qvalues.append(q_values.max())
            self._stats_stddev_qvalues.append(np.std(q_values))

        self._tracker = Tracker(new_state, new_action)
        self._actions_taken += 1

        return new_action

    def observe(self, old_state, action, reward, new_state, is_terminal):
        if is_terminal:
            self._history.reset()

        min_val, max_val = self._reward_clipping
        reward = max(min_val, min(max_val, reward))
        self._memory.append(old_state, int(action), reward, is_terminal)

    def learn(self):
        if (self._actions_taken % self._train_frequency) == 0:
            minibatch = self._memory.minibatch(self._minibatch_size)
            q_t_target = self._compute_q(*minibatch)

            self._model.train(minibatch[0], q_t_target, minibatch[1])
            self._stats_loss.append(self._model.loss_val)

    def inject_summaries(self, idx):
        if len(self._stats_mean_qvalues) > 0:
            self.visualize(idx, "%s/episode mean q" % self.name,
                           np.asscalar(np.mean(self._stats_mean_qvalues)))
            self.visualize(idx, "%s/episode mean stddev.q" % self.name,
                           np.asscalar(np.mean(self._stats_stddev_qvalues)))

        if len(self._stats_loss) > 0:
            self.visualize(idx, "%s/episode mean loss" % self.name,
                           np.asscalar(np.mean(self._stats_loss)))

        if len(self._stats_rewards) > 0:
            self.visualize(idx, "%s/episode mean reward" % self.name,
                           np.asscalar(np.mean(self._stats_rewards)))

            # Reset
            self._stats_mean_qvalues = []
            self._stats_stddev_qvalues = []
            self._stats_loss = []
            self._stats_rewards = []

    def _compute_q(self, pres, actions, posts, rewards, terminals):
        """ Compute the Q Values from input states """

        q_hat = self._model.evaluate(posts, model=QModel.TARGET_NETWORK)
        q_hat_eval = q_hat[np.arange(len(actions)), q_hat.argmax(axis=1)]

        q_targets = (1 - terminals) * (self._gamma * q_hat_eval) + rewards
        return np.array(q_targets, dtype=np.float32)
