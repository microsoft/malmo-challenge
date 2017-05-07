import logging

import chainer
from chainer import functions as fun
from chainer import links
from chainerrl.recurrent import RecurrentChainMixin

logger = logging.getLogger(__name__)


class NNQFunc(chainer.Chain):
    def __init__(self, input_dim, output_dim, hidden_units, w_scale=0.1):
        super(NNQFunc, self).__init__(
            input_layer=links.Linear(input_dim, hidden_units, wscale=w_scale),
            mid_layer=links.Linear(hidden_units, hidden_units, wscale=w_scale),
            output_layer=links.Linear(hidden_units, output_dim, wscale=w_scale))

        self.h2 = None
        logger.log(msg='Initialized network {}'.format(self.__class__.__name__),
                   level=logging.INFO)

    def __call__(self, x, test=False):
        h1 = fun.relu(self.input_layer(x))
        self.h2 = fun.relu(self.mid_layer(h1))
        out = self.output_layer(self.h2)
        return out


class RecNNQFunc(chainer.Chain, RecurrentChainMixin):
    def __init__(self, input_dim, output_dim, hidden_units, rec_dim=5, w_scale=0.1):
        super(RecNNQFunc, self).__init__(
            input_layer=links.Linear(input_dim, hidden_units, wscale=w_scale),
            mid_layer=links.Linear(hidden_units, hidden_units, wscale=w_scale),
            mid_rec_layer=links.LSTM(input_dim, rec_dim, forget_bias_init=1.,
                                     lateral_init=w_scale,
                                     upward_init=w_scale),
            output_layer=links.Linear(rec_dim + hidden_units, output_dim, wscale=w_scale))

        self.rec_h1 = None
        self.h2 = None
        logger.log(msg='Initialized network {}'.format(self.__class__.__name__),
                   level=logging.INFO)

    def __call__(self, x, test=False):
        h1 = fun.relu(self.input_layer(x))
        self.rec_h1 = self.mid_rec_layer(x)
        self.h2 = fun.relu(self.mid_layer(h1))
        self.h3 = fun.concat((self.h2, self.rec_h1))
        out = self.output_layer(self.h3)
        return out
