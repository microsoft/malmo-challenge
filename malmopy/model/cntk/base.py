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

import numpy as np
from cntk.device import cpu, gpu, try_set_default_device
from cntk.train.distributed import Communicator
from cntk.learners import set_default_unit_gain_value
from cntk.ops import abs, element_select, less, square, sqrt, reduce_sum, reduce_mean

from ...visualization import Visualizable


def rmse(y, y_hat, axis=0):
    """
     Compute the Root Mean Squared error as part of the model graph

     :param y: CNTK Variable holding the true value of Y
     :param y_hat: CNTK variable holding the estimated value of Y
     :param axis: The axis over which to compute the mean, 0 by default
     :return: Root Mean Squared error
     """
    return sqrt(reduce_mean(square(y_hat - y), axis=axis))


def huber_loss(y_hat, y, delta):
    """
    Compute the Huber Loss as part of the model graph

    Huber Loss is more robust to outliers. It is defined as:
     if |y - h_hat| < delta :
        0.5 * (y - y_hat)**2
    else :
        delta * |y - y_hat| - 0.5 * delta**2

    :param y: Target value
    :param y_hat: Estimated value
    :param delta: Outliers threshold
    :return: float
    """
    half_delta_squared = 0.5 * delta * delta
    error = y - y_hat
    abs_error = abs(error)

    less_than = 0.5 * square(error)
    more_than = (delta * abs_error) - half_delta_squared

    loss_per_sample = element_select(less(abs_error, delta), less_than, more_than)

    return reduce_sum(loss_per_sample, name='loss')


def as_learning_rate_by_sample(learning_rate_per_minibatch, minibatch_size, momentum=0, momentum_as_unit_gain=False):
    """
    Compute the scale parameter for the learning rate to match the learning rate
    definition used in other deep learning frameworks.
    In CNTK, gradients are calculated as follows:
        g(t + 1) = momentum * v(t) + (1-momentum) * gradient(t)

    Whereas in other frameworks they are computed this way :
        g(t + 1) = momentum * v(t)

    According to the above equations we need to scale the learning rate with regard to the momentum by a
    factor of 1/(1 - momentum)
    :param learning_rate_per_minibatch: The current learning rate
    :param minibatch_size: Size of the minibatch
    :param momentum: The current momentum (0 by default, used only when momentum_as_unit_gain is True)
    :param momentum_as_unit_gain: Indicate whetherf the momentum is a unit gain factor (CNTK) or not (TensorFlow, etc.)
    :return: Scaled learning rate according to momentum and minibatch size
    """
    assert learning_rate_per_minibatch > 0, "learning_rate_per_minibatch cannot be < 0"
    assert minibatch_size > 0, "minibatch_size cannot be < 1"

    learning_rate_per_sample = learning_rate_per_minibatch / minibatch_size

    if momentum_as_unit_gain:
        learning_rate_per_sample /= (1. - momentum)

    return learning_rate_per_sample


def as_momentum_as_time_constant(momentum, minibatch_size):
    """ Convert a momentum  provided a global for the a full minibatch
    to the momentum as number of sample seen rate by sample

    momentum_as_time_constant = -minibatch_size / (np.log(momentum))
    """
    return np.ceil(-minibatch_size / (np.log(momentum)))


def prepend_batch_seq_axis(tensor):
    """
    CNTK uses 2 dynamic axes (batch, sequence, input_shape...).
    To have a single sample with length 1 you need to pass (1, 1, input_shape...)
    This method reshapes a tensor to add to the batch and sequence axis equal to 1.
    :param tensor: The tensor to be reshaped
    :return: Reshaped tensor with batch and sequence axis = 1
    """
    return tensor.reshape((1, 1,) + tensor.shape)


def prepend_batch_axis(tensor):
    """
    CNTK uses 2 dynamic axes (batch, sequence, input_shape...).
    If you define variables with dynamic_axes=[Axis.default_batch_axis()] you can rid of sequence axis

    To have a single sample with length 1 you need to pass (1, input_shape...)
    This method reshapes a tensor to add to the batch and sequence axis equal to 1.
    :param tensor: The tensor to be reshaped
    :return: Reshaped tensor with batch and sequence axis = 1
    """
    return tensor.reshape((1,) + tensor.shape)


class CntkModel(Visualizable):
    """ Base class for CNTK based neural networks.

    It handles the management of the CPU/GPU device and provides commodity methods for exporting the model
    """

    def __init__(self, device_id=None, unit_gain=False, n_workers=1, visualizer=None):
        """
        Abstract constructor of CNTK model.
        This constructor wraps CNTK intialization and tuning
        :param device_id: Use None if you want CNTK to use the best available device, -1 for CPU, >= 0 for GPU
        :param n_workers: Number of concurrent workers for distributed training. Keep set to 1 for non distributed mode
        :param visualizer: Optional visualizer allowing model to save summary data
        """
        assert n_workers >= 1, 'n_workers should be at least 1 (not distributed) or > 1 if distributed'

        Visualizable.__init__(self, visualizer)

        self._model = None
        self._learner = None
        self._loss = None
        self._distributed = n_workers > 1

        if isinstance(device_id, int):
            try_set_default_device(cpu() if device_id == -1 else gpu(device_id))

        set_default_unit_gain_value(unit_gain)

    def _build_model(self):
        raise NotImplementedError()

    @property
    def loss_val(self):
        raise NotImplementedError()

    @property
    def model(self):
        return self._model

    @property
    def distributed_training(self):
        return self._distributed

    @property
    def distributed_rank(self):
        if self._distributed:
            if self._learner and hasattr(self._learner, 'communicator'):
                return self._learner.communicator().rank()
            else:
                return 0

    def load(self, input_file):
        if self._model is None:
            raise ValueError("cannot load to a model that equals None")

        self._model.restore(input_file)

    def save(self, output_file):
        if self._model is None:
            raise ValueError("cannot save a model that equals None")

        self._model.save(output_file)

    def finalize(self):
        if self._distributed:
            Communicator.finalize()
