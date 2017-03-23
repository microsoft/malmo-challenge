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


class BaseExplorer:
    """ Explore/exploit logic wrapper"""

    def __call__(self, step, nb_actions):
        return self.explore(step, nb_actions)

    def is_exploring(self, step):
        """ Returns True when exploring, False when exploiting """
        raise NotImplementedError()

    def explore(self, step, nb_actions):
        """ Generate an exploratory action """
        raise NotImplementedError()


class LinearEpsilonGreedyExplorer(BaseExplorer):
    """ Explore/exploit logic wrapper


    This implementation uses linear interpolation between
    epsilon_max and epsilon_min to generate linearly anneal epsilon as a function of the current episode.

    3 cases exists:
        - If 0 <= episode < eps_min_time then epsilon = interpolator(episode)
        - If episode >= eps_min_time then epsilon then epsilon = eps_min
        - Otherwise epsilon = eps_max
    """

    def __init__(self, eps_max, eps_min, eps_min_time):
        assert eps_max > eps_min
        assert eps_min_time > 0

        self._eps_min_time = eps_min_time
        self._eps_min = eps_min
        self._eps_max = eps_max

        self._a = -(eps_max - eps_min) / eps_min_time

    def _epsilon(self, step):
        if step < 0:
            return self._eps_max
        elif step > self._eps_min_time:
            return self._eps_min
        else:
            return self._a * step + self._eps_max

    def is_exploring(self, step):
        return np.random.rand() < self._epsilon(step)

    def explore(self, step, nb_actions):
        return np.random.randint(0, nb_actions)
