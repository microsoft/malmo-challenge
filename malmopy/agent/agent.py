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

import os
import sys
from collections import Iterable

import numpy as np

from ..visualization import Visualizable


class BaseAgent(Visualizable):
    """
    Represents an agent that interacts with an environment
    """

    def __init__(self, name, nb_actions, visualizer=None):
        assert nb_actions > 0, 'Agent should at least have 1 action (got %d)' % nb_actions

        super(BaseAgent, self).__init__(visualizer)

        self.name = name
        self.nb_actions = nb_actions

    def act(self, new_state, reward, done, is_training=False):
        raise NotImplementedError()

    def save(self, out_dir):
        pass

    def load(self, out_dir):
        pass

    def inject_summaries(self, idx):
        pass


class RandomAgent(BaseAgent):
    """
    An agent that selects actions uniformly at random
    """

    def __init__(self, name, nb_actions, delay_between_action=0, visualizer=None):
        super(RandomAgent, self).__init__(name, nb_actions, visualizer)

        self._delay = delay_between_action

    def act(self, new_state, reward, done, is_training=False):
        if self._delay > 0:
            from time import sleep
            sleep(self._delay)

        return np.random.randint(0, self.nb_actions)


class ConsoleAgent(BaseAgent):
    """ Provide a console interface for mediating human interaction with
   an environment

    Users are prompted for input when an action is required:

    Agent-1, what do you want to do?
        1: action1
        2: action2
        3: action3
        ...
        N: actionN
    Agent-1: 2
    ...
    """

    def __init__(self, name, actions, stdin=None):
        assert isinstance(actions, Iterable), 'actions need to be iterable (e.g., list, tuple)'
        assert len(actions) > 0, 'actions need at least one element'

        super(ConsoleAgent, self).__init__(name, len(actions))

        self._actions = actions

        if stdin is not None:
            sys.stdin = os.fdopen(stdin)

    def act(self, new_state, reward, done, is_training=False):
        user_action = None

        while user_action is None:
            self._print_choices()
            try:
                user_input = input("%s: " % self.name)
                user_action = int(user_input)
                if user_action < 0 or user_action > len(self._actions) - 1:
                    user_action = None
                    print("Provided input is not valid should be [0, %d]" % (len(self._actions) - 1))
            except ValueError:
                user_action = None
                print("Provided input is not valid should be [0, %d]" % (len(self._actions) - 1))

        return user_action

    def _print_choices(self):
        print("\n%s What do you want to do?" % self.name)

        for idx, action in enumerate(self._actions):
            print("\t%d : %s" % (idx, action))


class ReplayMemory(object):
    """
    Simple representation of agent memory
    """

    def __init__(self, max_size, state_shape):
        assert max_size > 0, 'size should be > 0 (got %d)' % max_size

        self._pos = 0
        self._count = 0
        self._max_size = max_size
        self._state_shape = state_shape
        self._states = np.empty((max_size,) + state_shape, dtype=np.float32)
        self._actions = np.empty(max_size, dtype=np.uint8)
        self._rewards = np.empty(max_size, dtype=np.float32)
        self._terminals = np.empty(max_size, dtype=np.bool)

    def append(self, state, action, reward, is_terminal):
        """
        Appends the specified memory to the history.
        :param state: The state to append (should have the same shape as defined at initialization time)
        :param action: An integer representing the action done
        :param reward: An integer reprensenting the reward received for doing this action
        :param is_terminal: A boolean specifying if this state is a terminal (episode has finished)
        :return:
        """
        assert state.shape == self._state_shape, \
            'Invalid state shape (required: %s, got: %s)' % (self._state_shape, state.shape)

        self._states[self._pos, ...] = state
        self._actions[self._pos] = action
        self._rewards[self._pos] = reward
        self._terminals[self._pos] = is_terminal

        self._count = max(self._count, self._pos + 1)
        self._pos = (self._pos + 1) % self._max_size

    def __len__(self):
        """
        Number of elements currently stored in the memory (same as #size())
        See #size()
        :return: Integer : max_size >= size() >= 0
        """
        return self.size

    @property
    def last(self):
        """
        Return the last observation from the memory
        :return: Tuple (state, action, reward, terminal)
        """
        idx = self._pos
        return self._states[idx], self._actions[idx], self._rewards[idx], self._terminals[idx]

    @property
    def size(self):
        """
        Number of elements currently stored in the memory
        :return: Integer : max_size >= size >= 0
        """
        return self._count

    @property
    def max_size(self):
        """
        Maximum number of elements that can fit in the memory
        :return: Integer > 0
        """
        return self._max_size

    @property
    def history_length(self):
        """
        Number of states stacked along the first axis
        :return: int >= 1
        """
        return 1

    def sample(self, size, replace=False):
        """
        Generate a random sample of desired size (if available) from the current memory
        :param size: Number of samples
        :param replace: True if sampling with replacement
        :return: Integer[size] representing the sampled indices
        """
        return np.random.choice(self._count, size, replace=replace)

    def get_state(self, index):
        """
        Return the specified state
        :param index: State's index
        :return: state : (input_shape)
        """
        index %= self.size
        return self._states[index]

    def get_action(self, index):
        """
        Return the specified action
        :param index: Action's index
        :return: Integer
        """
        index %= self.size
        return self._actions[index]

    def get_reward(self, index):
        """
        Return the specified reward
        :param index: Reward's index
        :return: Integer
        """
        index %= self.size
        return self._rewards[index]

    def minibatch(self, size):
        """
        Generate a minibatch with the number of samples specified by the size parameter.
        :param size: Minibatch size
        :return: Tensor[minibatch_size, input_shape...)
        """
        indexes = self.sample(size)

        pre_states = np.array([self.get_state(index) for index in indexes], dtype=np.float32)
        post_states = np.array([self.get_state(index + 1) for index in indexes], dtype=np.float32)
        actions = self._actions[indexes]
        rewards = self._rewards[indexes]
        terminals = self._terminals[indexes]

        return pre_states, actions, post_states, rewards, terminals

    def save(self, out_file):
        """
        Save the current memory into a file in Numpy format
        :param out_file: File storage path
        :return:
        """
        np.savez_compressed(out_file, states=self._states, actions=self._actions,
                            rewards=self._rewards, terminals=self._terminals)

    def load(self, in_dir):
        pass
