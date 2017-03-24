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

import chainer.cuda as cuda
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import ChainList
from chainer.initializers import HeUniform
from chainer.optimizers import Adam
from chainer.serializers import save_npz, load_npz

from ..model import QModel
from ...util import check_rank, get_rank


class ChainerModel(ChainList):
    """
    Wraps a Chainer Chain and enforces the model to be callable.
    Every model should override the __call__ method as a forward call.
    """

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

        super(ChainerModel, self).__init__(*self._build_model())

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def _build_model(self):
        raise NotImplementedError()


class MLPChain(ChainerModel):
    """
    Create a Multi Layer Perceptron neural network.
    The number of layers and units for each layer can be specified using hidden_layer_sizes.
    For example for a 128 units on the first hidden layer, then 256 on the second and 512 on the third:

    >>> MLPChain(input_shape=(28, 28), output_shape=10, hidden_layer_sizes=(128, 256, 512))

    Note : The network will contain len(hidden_layer_sizes) + 2 layers because 
    of the input layer and the output layer.
    """

    def __init__(self, in_shape, output_shape,
                 hidden_layer_sizes=(512, 512, 512), activation=F.relu):
        self._activation = activation
        self._hidden_layer_sizes = hidden_layer_sizes

        super(MLPChain, self).__init__(in_shape, output_shape)

    @property
    def hidden_layer_sizes(self):
        return self._hidden_layer_sizes

    def __call__(self, x):
        f = self._activation

        for layer in self[:-1]:
            x = f(layer(x))
        return self[-1](x)

    def _build_model(self):
        hidden_layers = [L.Linear(None, units) for units in
                         self._hidden_layer_sizes]
        hidden_layers += [L.Linear(None, self.output_shape)]

        return hidden_layers


class ReducedDQNChain(ChainerModel):
    """
    Simplified DQN topology:
    
    Convolution(64, kernel=(4, 4), strides=(2, 2)
    Convolution(64, kernel=(3, 3), strides=(1, 1)
    Dense(512)
    Dense(output_shape)
    """
    def __init__(self, in_shape, output_shape):
        super(ReducedDQNChain, self).__init__(in_shape, output_shape)

    def __call__(self, x):
        for layer in self[:-1]:
            x = F.relu(layer(x))
        return self[-1](x)

    def _build_model(self):
        initializer = HeUniform()
        in_shape = self.input_shape[0]

        return [L.Convolution2D(in_shape, 64, ksize=4, stride=2,
                                initialW=initializer),
                L.Convolution2D(64, 64, ksize=3, stride=1,
                                initialW=initializer),
                L.Linear(None, 512, initialW=HeUniform(0.1)),
                L.Linear(512, self.output_shape, initialW=HeUniform(0.1))]


class DQNChain(ChainerModel):
    """
    DQN topology as in 
    (Mnih & al. 2015): Human-level control through deep reinforcement learning"
    Nature 518.7540 (2015): 529-533.
    
    Convolution(32, kernel=(8, 8), strides=(4, 4)
    Convolution(64, kernel=(4, 4), strides=(2, 2)
    Convolution(64, kernel=(3, 3), strides=(1, 1)
    Dense(512)
    Dense(output_shape)
    """

    def __init__(self, in_shape, output_shape):
        super(DQNChain, self).__init__(in_shape, output_shape)

    def __call__(self, x):
        for layer in self[:-1]:
            x = F.relu(layer(x))
        return self[-1](x)

    def _build_model(self):
        initializer = HeUniform()
        in_shape = self.input_shape[0]

        return [L.Convolution2D(in_shape, 32, ksize=8, stride=4,
                                initialW=initializer),
                L.Convolution2D(32, 64, ksize=4, stride=2,
                                initialW=initializer),
                L.Convolution2D(64, 64, ksize=3, stride=1,
                                initialW=initializer),
                L.Linear(7 * 7 * 64, 512, initialW=HeUniform(0.01)),
                L.Linear(512, self.output_shape, initialW=HeUniform(0.01))]


class QNeuralNetwork(QModel):
    def __init__(self, model, target, device_id=-1,
                 learning_rate=0.00025, momentum=.9,
                 minibatch_size=32, update_interval=10000):

        assert isinstance(model, ChainerModel), \
            'model should inherit from ChainerModel'

        super(QNeuralNetwork, self).__init__(model.input_shape,
                                             model.output_shape)

        self._gpu_device = None
        self._loss_val = 0

        # Target model update method
        self._steps = 0
        self._target_update_interval = update_interval

        # Setup model and target network
        self._minibatch_size = minibatch_size
        self._model = model
        self._target = target
        self._target.copyparams(self._model)

        # If GPU move to GPU memory
        if device_id >= 0:
            with cuda.get_device(device_id) as device:
                self._gpu_device = device
                self._model.to_gpu(device)
                self._target.to_gpu(device)

        # Setup optimizer
        self._optimizer = Adam(learning_rate, momentum, 0.999)
        self._optimizer.setup(self._model)

    def evaluate(self, environment, model=QModel.ACTION_VALUE_NETWORK):
        if check_rank(environment.shape, get_rank(self._input_shape)):
            environment = environment.reshape((1,) + environment.shape)

        # Move data if necessary
        if self._gpu_device is not None:
            environment = cuda.to_gpu(environment, self._gpu_device)

        if model == QModel.ACTION_VALUE_NETWORK:
            output = self._model(environment)
        else:
            output = self._target(environment)

        return cuda.to_cpu(output.data)

    def train(self, x, y, actions=None):
        actions = actions.astype(np.int32)
        batch_size = len(actions)

        if self._gpu_device:
            x = cuda.to_gpu(x, self._gpu_device)
            y = cuda.to_gpu(y, self._gpu_device)
            actions = cuda.to_gpu(actions, self._gpu_device)

        q = self._model(x)
        q_subset = F.reshape(F.select_item(q, actions), (batch_size, 1))
        y = y.reshape(batch_size, 1)

        loss = F.huber_loss(q_subset, y, 1.0)

        self._model.cleargrads()
        loss.backward()
        self._optimizer.update()

        self._loss_val = np.asscalar(cuda.to_cpu(loss.data))

        # Keeps track of the number of train() calls
        self._steps += 1
        if self._steps % self._target_update_interval == 0:
            # copy weights
            self._target.copyparams(self._model)

    @property
    def loss_val(self):
        return self._loss_val  # / self._minibatch_size

    def save(self, output_file):
        save_npz(output_file, self._model)

    def load(self, input_file):
        load_npz(input_file, self._model)

        # Copy parameter from model to target
        self._target.copyparams(self._model)
