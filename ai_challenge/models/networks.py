import logging

import chainer
from chainer import functions as fun
from chainer import links
from chainerrl.recurrent import RecurrentChainMixin
from chainerrl.agents import a3c
from chainerrl.policies import SoftmaxPolicy

logger = logging.getLogger(__name__)


class NNQFunc(chainer.Chain):
    def __init__(self, input_dim, output_dim, hidden_units, w_scale=0.1, activation='relu'):
        super(NNQFunc, self).__init__(
            input_layer=links.Linear(input_dim, hidden_units, wscale=w_scale),
            mid_layer=links.Linear(hidden_units, hidden_units, wscale=w_scale),
            output_layer=links.Linear(hidden_units, output_dim, wscale=w_scale))

        self.h2 = None
        logger.log(msg='Initialized network {}.'.format(self.__class__.__name__),
                   level=logging.INFO)
        self.activation = getattr(fun, activation)

    def __call__(self, x, test=False):
        h1 = self.activation(self.input_layer(x))
        self.h2 = self.activation(self.mid_layer(h1))
        out = self.output_layer(self.h2)
        return out


class RecNNQFunc(chainer.Chain, RecurrentChainMixin):
    def __init__(self, input_dim, output_dim, hidden_units, rec_dim=10, w_scale=0.1,
                 activation='relu'):
        super(RecNNQFunc, self).__init__(
            input_layer=links.Linear(input_dim, hidden_units, wscale=w_scale),
            mid_layer=links.Linear(hidden_units, hidden_units, wscale=w_scale),
            mid_rec_layer=links.LSTM(input_dim, rec_dim, forget_bias_init=1.,
                                     lateral_init=w_scale,
                                     upward_init=w_scale),
            output_layer=links.Linear(rec_dim + hidden_units, output_dim, wscale=w_scale))

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
        self.pi = SoftmaxPolicy(model=RecNNQFunc(input_dim, output_dim, hidden_units))
        self.v = RecNNQFunc(input_dim, 1, hidden_units)
        super(A3CRecNN, self).__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)
