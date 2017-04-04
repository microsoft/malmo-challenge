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

from cntk import Value
from cntk.axis import Axis
from cntk.initializer import he_uniform, he_normal
from cntk.layers import Convolution, Dense, default_options
from cntk.layers.higher_order_layers import Sequential
from cntk.learners import adam, momentum_schedule, learning_rate_schedule, \
    UnitType
from cntk.ops import input, relu, reduce_sum
from cntk.ops.functions import CloneMethod
from cntk.train.trainer import Trainer

from . import CntkModel, prepend_batch_axis, huber_loss
from ..model import QModel
from ...util import check_rank


class QNeuralNetwork(CntkModel, QModel):
    """
    Represents a learning capable entity using CNTK
    """

    def __init__(self, in_shape, output_shape, device_id=None,
                 learning_rate=0.00025, momentum=0.9,
                 minibatch_size=32, update_interval=10000,
                 n_workers=1, visualizer=None):

        """
        Q Neural Network following Mnih and al. implementation and default options.

        The network has the following topology:
        Convolution(32, (8, 8))
        Convolution(64, (4, 4))
        Convolution(64, (2, 2))
        Dense(512)

        :param in_shape: Shape of the observations perceived by the learner (the neural net input)
        :param output_shape: Size of the action space (mapped to the number of output neurons)

        :param device_id: Use None to let CNTK select the best available device,
                          -1 for CPU, >= 0 for GPU
                          (default: None)

        :param learning_rate: Learning rate
                              (default: 0.00025, as per Mnih et al.)

        :param momentum: Momentum, provided as momentum value for
                         averaging gradients without unit gain filter
                         Note that CNTK does not currently provide an implementation
                         of Graves' RmsProp with momentum.
                         It uses AdamSGD optimizer instead.
                         (default: 0, no momentum with RProp optimizer)

        :param minibatch_size: Minibatch size
                               (default: 32, as per Mnih et al.)

        :param n_workers: Number of concurrent worker for distributed training.
                          (default: 1, not distributed)

        :param visualizer: Optional visualizer allowing the model to save summary data
                           (default: None, no visualization)

        Ref: Mnih et al.: "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.
        """

        assert learning_rate > 0, 'learning_rate should be > 0'
        assert 0. <= momentum < 1, 'momentum should be 0 <= momentum < 1'

        QModel.__init__(self, in_shape, output_shape)
        CntkModel.__init__(self, device_id, False, n_workers, visualizer)

        self._nb_actions = output_shape
        self._steps = 0
        self._target_update_interval = update_interval
        self._target = None

        # Input vars
        self._environment = input(in_shape, name='env',
                                  dynamic_axes=(Axis.default_batch_axis()))
        self._q_targets = input(1, name='q_targets',
                                dynamic_axes=(Axis.default_batch_axis()))
        self._actions = input(output_shape, name='actions',
                              dynamic_axes=(Axis.default_batch_axis()))

        # Define the neural network graph
        self._model = self._build_model()(self._environment)
        self._target = self._model.clone(
            CloneMethod.freeze, {self._environment: self._environment}
        )

        # Define the learning rate
        lr_schedule = learning_rate_schedule(learning_rate, UnitType.minibatch)

        # AdamSGD optimizer
        m_schedule = momentum_schedule(momentum)
        vm_schedule = momentum_schedule(0.999)
        l_sgd = adam(self._model.parameters, lr_schedule,
                     momentum=m_schedule,
                     unit_gain=True,
                     variance_momentum=vm_schedule)

        if self.distributed_training:
            raise NotImplementedError('ASGD not implemented yet.')

        # _actions is a sparse 1-hot encoding of the actions done by the agent
        q_acted = reduce_sum(self._model * self._actions, axis=0)

        # Define the trainer with Huber Loss function
        criterion = huber_loss(q_acted, self._q_targets, 1.0)

        self._learner = l_sgd
        self._trainer = Trainer(self._model, (criterion, None), l_sgd)

    @property
    def loss_val(self):
        return self._trainer.previous_minibatch_loss_average

    def _build_model(self):
        with default_options(init=he_uniform(), activation=relu, bias=True):
            model = Sequential([
                Convolution((8, 8), 32, strides=(4, 4)),
                Convolution((4, 4), 64, strides=(2, 2)),
                Convolution((3, 3), 64, strides=(1, 1)),
                Dense(512, init=he_normal(0.01)),
                Dense(self._nb_actions, activation=None, init=he_normal(0.01))
            ])
            return model

    def train(self, x, q_value_targets, actions=None):
        assert actions is not None, 'actions cannot be None'

        # We need to add extra dimensions to shape [N, 1] => [N, 1]
        if check_rank(q_value_targets.shape, 1):
            q_value_targets = q_value_targets.reshape((-1, 1))

        # Add extra dimensions to match shape [N, 1] required by one_hot
        if check_rank(actions.shape, 1):
            actions = actions.reshape((-1, 1))

        # We need batch axis
        if check_rank(x.shape, len(self._environment.shape)):
            x = prepend_batch_axis(x)

        self._trainer.train_minibatch({
            self._environment: x,
            self._actions: Value.one_hot(actions, self._nb_actions),
            self._q_targets: q_value_targets
        })

        # Counter number of train calls
        self._steps += 1

        # Update the model with the target one
        if (self._steps % self._target_update_interval) == 0:
            self._target = self._model.clone(
                CloneMethod.freeze, {self._environment: self._environment}
            )

    def evaluate(self, data, model=QModel.ACTION_VALUE_NETWORK):
        # If evaluating a single sample, expand the minibatch axis
        # (minibatch = 1, input_shape...)
        if len(data.shape) == len(self.input_shape):
            data = prepend_batch_axis(data)  # Append minibatch dim

        if model == QModel.TARGET_NETWORK:
            predictions = self._target.eval({self._environment: data})
        else:
            predictions = self._model.eval({self._environment: data})
        return predictions.squeeze()


class ReducedQNeuralNetwork(QNeuralNetwork):
    """
    Represents a learning capable entity using CNTK, reduced model
    """

    def __init__(self, in_shape, output_shape, device_id=None,
                 learning_rate=0.00025, momentum=0.9,
                 minibatch_size=32, update_interval=10000,
                 n_workers=1, visualizer=None):

        QNeuralNetwork.__init__(self, in_shape, output_shape, device_id,
                                learning_rate, momentum, minibatch_size, update_interval,
                                n_workers, visualizer)

    def _build_model(self):
        with default_options(init=he_uniform(), activation=relu, bias=True):
            model = Sequential([
                Convolution((4, 4), 64, strides=(2, 2), name='conv1'),
                Convolution((3, 3), 64, strides=(1, 1), name='conv2'),
                Dense(512, name='dense1', init=he_normal(0.01)),
                Dense(self._nb_actions, activation=None, init=he_normal(0.01), name='qvalues')
            ])
            return model