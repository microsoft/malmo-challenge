"""
Various architectures of NN.
"""
import logging

import chainer
from chainer import functions as fun
from chainer import links
from chainerrl.recurrent import RecurrentChainMixin
from chainerrl.agents import a3c
from chainerrl.policies import SoftmaxPolicy
from chainerrl.q_function import StateQFunction

logger = logging.getLogger(__name__)


class NN(chainer.Chain):
    def __init__(self, input_dim, output_dim, hidden_units, w_scale=0.01, activation='relu'):
        """
        Creates a basic NN with 2 hidden layers.
        :param input_dim: type int, the dimension of input
        :param output_dim: type int, the dimension of output
        :param hidden_units: type int, the dimension of hidden layers
        :param w_scale: type float, the scale used to initialize the weights, look
        at Chainer initialization
        :param activation: type str, the activation used in network
        """
        super(NN, self).__init__(
            input_layer=links.Linear(input_dim, hidden_units, wscale=w_scale),
            mid_layer=links.Linear(hidden_units, hidden_units, wscale=w_scale),
            output_layer=links.Linear(hidden_units, output_dim, wscale=w_scale))
        logger.log(msg='Initialized network {}.'.format(self.__class__.__name__),
                   level=logging.INFO)
        self.activation = getattr(fun, activation)

    def __call__(self, x, test=False):
        """
        Forward pass through the network.
        :param x: type numpy array, the input to the network
        :param test: type bool, true if network is in testing mode
        :return: type numpy array, the output from network
        """
        h1 = self.activation(self.input_layer(x))
        h2 = self.activation(self.mid_layer(h1))
        out = self.output_layer(h2)
        return out


class RecNN(chainer.Chain, RecurrentChainMixin):
    def __init__(self, input_dim, output_dim, hidden_units, w_scale=0.01, activation='relu'):
        """
        Creates a basic recurrent NN.
        :param input_dim: type int, the dimension of input
        :param output_dim: type int, the dimension of output
        :param hidden_units: type int, the dimension of hidden layers
        :param w_scale: type float, the scale used to initialize the weights, look
        at Chainer initialization
        :param activation: type str, the activation used in network
        """
        super(RecNN, self).__init__(
            input_layer=links.Linear(input_dim, hidden_units, wscale=w_scale,
                                     bias=initial_bias),
            mid_rec_layer=links.LSTM(hidden_units, hidden_units,
                                     forget_bias_init=0.,
                                     lateral_init=w_scale,
                                     upward_init=w_scale),
            output_layer=links.Linear(hidden_units, output_dim, wscale=w_scale))

        self.lstm_c = self['mid_rec_layer'].c
        self.lstm_h = self['mid_rec_layer'].h
        logger.log(msg='Initialized network {}.'.format(self.__class__.__name__),
                   level=logging.INFO)
        self.activation = getattr(fun, activation)

    def __call__(self, x, test=False):
        """
        Forward pass through the network.
        :param x: type numpy array, the input to the network
        :param test: type bool, true if network is in testing mode
        :return: type numpy array, the output from network
        """
        h1 = self.activation(self.input_layer(x))
        h2 = self.activation(self.mid_rec_layer(h1))
        out = self.output_layer(h2)
        return out


class A3CRecNN(chainer.ChainList, a3c.A3CModel, RecurrentChainMixin):
    """
    Recurrent NN that can be used to train A3C.
    """
    def __init__(self, input_dim, output_dim, hidden_units):
        self.pi = SoftmaxPolicy(model=RecNN(input_dim, output_dim, hidden_units))
        self.v = RecNN(input_dim, 1, hidden_units)
        super(A3CRecNN, self).__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)


class A3CNN(chainer.ChainList, a3c.A3CModel):
    """
    NN that can be used to train A3C.
    """

    def __init__(self, input_dim, output_dim, hidden_units):
        self.pi = SoftmaxPolicy(model=NN(input_dim, output_dim, hidden_units))
        self.v = NN(input_dim, 1, hidden_units)
        super(A3CNN, self).__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)


class DuelingNN(chainer.Chain, StateQFunction):
    """
    Dueling architecture that can be used for Q learning from chainerrl.
    """

    def __init__(self, input_dim, output_dim, hidden_units, w_scale=0.01, activation='relu'):
        """
        Creates a dueling NN.
        :param input_dim: type int, the dimension of input
        :param output_dim: type int, the dimension of output
        :param hidden_units: type int, the dimension of hidden layers
        :param w_scale: type float, the scale used to initialize the weights, look
        at Chainer initialization
        :param activation: type str, the activation used in network
        """
        self.n_actions = output_dim
        self.activation = getattr(fun, activation)
        hidden = NN(input_dim, hidden_units, hidden_units, w_scale=w_scale)
        a_stream = links.Linear(hidden_units, output_dim, wscale=w_scale)
        v_stream = links.Linear(hidden_units, 1, wscale=w_scale)
        super(DuelingNN, self).__init__(hidden=hidden,
                                        a_stream=a_stream,
                                        v_stream=v_stream)
        self.h = None

    def __call__(self, x, test=False):
        """
        Forward pass through the network.
        :param x: type numpy array, the input to the network
        :param test: type bool, true if network is in testing mode
        :return: type numpy array, the output from network
        """
        self.h = self.activation(self.hidden(x))
        batch_size = x.shape[0]
        ya = self.a_stream(self.h)
        mean = fun.reshape(fun.sum(ya, axis=1) / self.n_actions, (batch_size, 1))
        ya, mean = fun.broadcast(ya, mean)
        ya -= mean
        ys = self.v_stream(self.h)
        ya, ys = fun.broadcast(ya, ys)
        q = ya + ys
        return q
