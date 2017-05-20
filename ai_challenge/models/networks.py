import logging

import chainer
from chainer import functions as fun
from chainer import links
from chainerrl.recurrent import RecurrentChainMixin
from chainerrl.agents import a3c
from chainerrl.policies import SoftmaxPolicy
from chainerrl.action_value import DiscreteActionValue
from chainerrl.q_function import StateQFunction

logger = logging.getLogger(__name__)

"""
Various architectures of NN.
"""


class NN(chainer.Chain):
    def __init__(self, input_dim, output_dim, hidden_units, w_scale=0.01, initial_bias=0.0,
                 activation='relu'):
        super(NN, self).__init__(
            input_layer=links.Linear(input_dim, hidden_units, wscale=w_scale,
                                     initial_bias=initial_bias),
            mid_layer=links.Linear(hidden_units, hidden_units, wscale=w_scale,
                                   initial_bias=initial_bias),
            output_layer=links.Linear(hidden_units, output_dim, wscale=w_scale,
                                      initial_bias=initial_bias))

        self.h2 = None
        logger.log(msg='Initialized network {}.'.format(self.__class__.__name__),
                   level=logging.INFO)
        self.activation = getattr(fun, activation)

    def __call__(self, x, test=False):
        h1 = self.activation(self.input_layer(x))
        self.h2 = self.activation(self.mid_layer(h1))
        out = self.output_layer(self.h2)
        return out


class RecNN(chainer.Chain, RecurrentChainMixin):
    def __init__(self, input_dim, output_dim, hidden_units, rec_dim=20, w_scale=0.01,
                 initial_bias=0.0,
                 activation='relu'):
        super(RecNN, self).__init__(
            input_layer=links.Linear(input_dim, hidden_units, wscale=w_scale,
                                     bias=initial_bias),
            mid_layer=links.Linear(hidden_units, hidden_units, wscale=w_scale,
                                   bias=initial_bias),
            mid_rec_layer=links.LSTM(input_dim, rec_dim,
                                     forget_bias_init=1.,
                                     lateral_init=w_scale,
                                     upward_init=w_scale),
            output_layer=links.Linear(rec_dim + hidden_units, output_dim, wscale=w_scale,
                                      bias=initial_bias))

        self.rec_h1 = None
        self.h2 = None
        self.h3 = None
        self.lstm_c = self['mid_rec_layer'].c
        self.lstm_h = self['mid_rec_layer'].h
        logger.log(msg='Initialized network {}.'.format(self.__class__.__name__),
                   level=logging.INFO)
        self.activation = getattr(fun, activation)

    def __call__(self, x, test=False):
        h1 = self.activation(self.input_layer(x))
        self.rec_h1 = self.mid_rec_layer(x)
        self.h2 = self.activation(self.mid_layer(h1))
        self.h3 = fun.concat((self.h2, self.rec_h1))
        out = self.output_layer(self.h3)
        return out


class A3CRecNN(chainer.ChainList, a3c.A3CModel, RecurrentChainMixin):
    def __init__(self, input_dim, output_dim, hidden_units):
        self.pi = SoftmaxPolicy(model=RecNN(input_dim, output_dim, hidden_units))
        self.v = RecNN(input_dim, 1, hidden_units)
        super(A3CRecNN, self).__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)


class A3CNN(chainer.ChainList, a3c.A3CModel, RecurrentChainMixin):
    def __init__(self, input_dim, output_dim, hidden_units):
        self.pi = SoftmaxPolicy(model=NN(input_dim, output_dim, hidden_units))
        self.v = NN(input_dim, 1, hidden_units)
        super(A3CNN, self).__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)


class DuelingNN(chainer.Chain, StateQFunction):
    def __init__(self, input_dim, output_dim, hidden_units, w_scale=0.01, activation='relu'):
        self.n_actions = output_dim
        self.activation = activation

        hidden = NN(input_dim, hidden_units, hidden_units, w_scale=w_scale)

        a_stream = links.Linear(hidden_units, output_dim, wscale=w_scale)
        v_stream = links.Linear(hidden_units, 1, wscale=w_scale)

        super(DuelingNN, self).__init__(hidden=hidden,
                                        a_stream=a_stream,
                                        v_stream=v_stream)

    def __call__(self, x, test=False):
        h = self.hidden(x)
        # Advantage
        batch_size = x.shape[0]
        ya = self.a_stream(h)
        mean = fun.reshape(fun.sum(ya, axis=1) / self.n_actions, (batch_size, 1))
        ya, mean = fun.broadcast(ya, mean)
        ya -= mean
        # State value
        ys = self.v_stream(h)
        ya, ys = fun.broadcast(ya, ys)
        q = ya + ys
        return q


class DuelingRecNN(chainer.Chain, StateQFunction, RecurrentChainMixin):
    def __init__(self, input_dim, output_dim, hidden_units, w_scale=0.01, activation='relu',
                 rec_dim=20):
        self.n_actions = output_dim
        self.activation = activation
        hidden = RecNN(input_dim, hidden_units, hidden_units, w_scale=w_scale, rec_dim=rec_dim)
        a_stream = links.Linear(hidden_units, output_dim, wscale=w_scale)
        v_stream = links.Linear(hidden_units, 1, wscale=w_scale)
        super(DuelingRecNN, self).__init__(hidden=hidden,
                                           a_stream=a_stream,
                                           v_stream=v_stream)

    def __call__(self, x, test=False):
        h = self.hidden(x)
        # Advantage
        batch_size = x.shape[0]
        ya = self.a_stream(h)
        mean = fun.reshape(fun.sum(ya, axis=1) / self.n_actions, (batch_size, 1))
        ya, mean = fun.broadcast(ya, mean)
        ya -= mean
        # State value
        ys = self.v_stream(h)
        ya, ys = fun.broadcast(ya, ys)
        q = ya + ys
        return q
