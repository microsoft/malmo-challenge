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

import numpy as np

from ..util import check_rank, get_rank, resize, rgb2gray


class StateBuilder(object):
    """
    StateBuilder are object that map environment state into another representation.

    Subclasses should override the build() method which can map specific environment behavior.
    For concrete examples, malmo package has some predefined state builder specific to Malmo
    """

    def build(self, environment):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.build(*args)


class ALEStateBuilder(StateBuilder):
    """
    Atari Environment state builder interface.

    This class assumes the environment.state() returns a numpy array.
    """

    SCALE_FACTOR = 1. / 255.

    def __init__(self, shape=(84, 84), normalize=True):
        self._shape = shape
        self._normalize = bool(normalize)

    def build(self, environment):
        if not isinstance(environment, np.ndarray):
            raise ValueError(
                'environment type is not a numpy.ndarray (got %s)' % str(
                    type(environment)))

        state = environment

        # Convert to gray
        if check_rank(environment.shape, 3):
            state = rgb2gray(environment)
        elif get_rank(state) > 3:
            raise ValueError('Cannot handle data with more than 3 dimensions')

        # Resize
        if state.shape != self._shape:
            state = resize(state, self._shape)

        return (state * ALEStateBuilder.SCALE_FACTOR).astype(np.float32)


class BaseEnvironment(object):
    """
    Abstract representation of an interactive environment
    """

    def __init__(self):
        self._score = 0.
        self._reward = 0.
        self._done = False
        self._state = None

    def do(self, action):
        """
        Do the specified action in the environment
        :param action: The action to be executed
        :return Tuple holding the new state, the reward and a flag indicating if the environment is done
        """
        raise NotImplementedError()

    def reset(self):
        """
        Reset the current environment's internal state.
        :return:
        """
        self._score = 0.
        self._reward = 0.
        self._done = False
        self._state = None

    @property
    def available_actions(self):
        """
        Returns the number of actions available in this environment
        :return: Integer > 0
        """
        raise NotImplementedError()

    @property
    def done(self):
        """
        Indicate if the current environment is in a terminal state
        :return: Boolean True if environment is in a terminal state, False otherwise
        """
        return self._done

    @property
    def state(self):
        """
        Return the current environment state
        :return:
        """
        return self._state

    @property
    def reward(self):
        """
        Return accumulated rewards
        :return: Float as the current accumulated rewards since last state
        """
        return self._reward

    @property
    def score(self):
        """
        Return the environment's current score.
        It is common that the score will the sum of observed rewards, but subclasses can change this behavior.
        :return: Number
        """
        return self._score

    @property
    def is_turn_based(self):
        """
        Indicate if this environment is running on a turn-based scenario (i.e.,
        agents take turns and wait for other agents' turns to complete before taking the next action).
        All subclasses should override this accordingly to the running scenario.
        As currently turn based is not the default behavior, the value returned is False
        :return: False
        """
        return False


class VideoCapableEnvironment(BaseEnvironment):
    """
    Represent the capacity of an environment to stream it's current state.
    Streaming relies on 2 properties :
     - fps : Number of frame this environment is able to generate each second
     - frame : The latest frame generated by this environment
    The display adapter should ask for a new frame with a 1/fps millisecond delay.
    If there is no updated frame, the frame property can return None.
    """

    def __init__(self):
        super(VideoCapableEnvironment, self).__init__()
        self._recording = False

    @property
    def recording(self):
        """
        Indicate if the current environment is dispatching the video stream
        :return: True if streaming, False otherwise
        """
        return self._recording

    @recording.setter
    def recording(self, val):
        """
        Change the internal recording state.
        :param val: True to activate video streaming, False otherwise
        :return:
        """
        self._recording = bool(val)

    @property
    def frame(self):
        """
        Return the most recent frame from the environment
        :return: PIL Image representing the current environment
        """
        raise NotImplementedError()
