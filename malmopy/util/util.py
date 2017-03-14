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
from math import sqrt

import numpy as np


def euclidean(a, b):
    assert len(a) == len(b), 'cannot compute distance when a and b have different shapes'
    return sqrt(sum([(a - b) ** 2 for a, b in zip(a, b)]))


def get_rank(x):
    """ Get a shape's rank """
    if isinstance(x, np.ndarray):
        return len(x.shape)
    elif isinstance(x, tuple):
        return len(x)
    else:
        return ValueError('Unable to determine rank of type: %s' % str(type(x)))


def check_rank(shape, required_rank):
    """ Check if the shape's rank equals the expected rank """
    if isinstance(shape, tuple):
        return len(shape) == required_rank
    else:
        return False


def isclose(a, b, atol=1e-01):
    """ Check if a and b are closer than tolerance level atol

    return abs(a - b) < atol
    """
    return abs(a - b) < atol


def ensure_path_exists(path):
    """ Ensure that the specified path exists on the filesystem """
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    if not os.path.exists(path):
        os.makedirs(path)
