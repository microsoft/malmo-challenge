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

import gym
import numpy as np
import six
from PIL import Image
from gym.wrappers import Monitor

from ..environment import VideoCapableEnvironment, StateBuilder, ALEStateBuilder


def need_record(episode_id):
    return episode_id % 1000 == 0


class GymEnvironment(VideoCapableEnvironment):
    """
    Wraps an Open AI Gym environment
    """

    def __init__(self, env_name, state_builder=ALEStateBuilder(), repeat_action=4, no_op=30, monitoring_path=None):
        assert isinstance(state_builder, StateBuilder), 'state_builder should inherit from StateBuilder'
        assert isinstance(repeat_action, (int, tuple)), 'repeat_action should be int or tuple'
        if isinstance(repeat_action, int):
            assert repeat_action >= 1, "repeat_action should be >= 1"
        elif isinstance(repeat_action, tuple):
            assert len(repeat_action) == 2, 'repeat_action should be a length-2 tuple: (min frameskip, max frameskip)'
            assert repeat_action[0] < repeat_action[1], 'repeat_action[0] should be < repeat_action[1]'

        super(GymEnvironment, self).__init__()

        self._state_builder = state_builder
        self._env = gym.make(env_name)
        self._env.frameskip = repeat_action
        self._no_op = max(0, no_op)
        self._done = True

        if monitoring_path is not None:
            self._env = Monitor(self._env, monitoring_path, video_callable=need_record)

    @property
    def available_actions(self):
        return self._env.action_space.n

    @property
    def state(self):
        return None if self._state is None else self._state_builder(self._state)

    @property
    def lives(self):
        return self._env.ale.lives()

    @property
    def frame(self):
        return Image.fromarray(self._state)

    def do(self, action):
        self._state, self._reward, self._done, _ = self._env.step(action)
        self._score += self._reward
        return self.state, self._reward, self._done

    def reset(self):
        super(GymEnvironment, self).reset()

        self._state = self._env.reset()

        # Random number of initial no-op to introduce stochasticity
        if self._no_op > 0:
            for _ in six.moves.range(np.random.randint(1, self._no_op)):
                self._state, _, _, _ = self._env.step(0)

        return self.state
