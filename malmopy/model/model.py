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


class BaseModel(object):
    """Represents a learning capable entity"""

    def __init__(self, in_shape, output_shape):
        self._input_shape = in_shape
        self._output_shape = output_shape

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

    @property
    def loss_val(self):
        raise NotImplementedError()

    def evaluate(self, environment):
        raise NotImplementedError()

    def train(self, x, y):
        raise NotImplementedError()

    def load(self, input_file):
        raise NotImplementedError()

    def save(self, output_file):
        raise NotImplementedError()


class QModel(BaseModel):
    ACTION_VALUE_NETWORK = 1 << 0
    TARGET_NETWORK = 1 << 1

    def evaluate(self, environment, model=ACTION_VALUE_NETWORK):
        raise NotImplementedError()

    def train(self, x, y, actions=None):
        raise NotImplementedError()


