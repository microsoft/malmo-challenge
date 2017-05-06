import chainer
from chainer import functions as fun
from chainer import links
from chainerrl.recurrent import RecurrentChainMixin


class NNQFunc(chainer.Chain):
    def __init__(self, input_dim, output_dim, hidden_units, w_scale=0.1):
        super(NNQFunc, self).__init__(
            input_layer=links.Linear(input_dim, hidden_units, wscale=w_scale),
            mid_layer=links.Linear(hidden_units, hidden_units, wscale=w_scale),
            output_layer=links.Linear(hidden_units, output_dim, wscale=w_scale))

    def __call__(self, x, test=False):
        h = self.input_layer(x)
        h = fun.relu(h)
        h = self.mid_layer(h)
        h = fun.relu(h)
        h = self.output_layer(h)
        return h


class RecNNQFunc(chainer.Chain, RecurrentChainMixin):
    def __init__(self, input_dim, output_dim, hidden_units, rec_dim=5, w_scale=0.1):
        super(RecNNQFunc, self).__init__(
            input_layer=links.Linear(input_dim, hidden_units, wscale=w_scale),
            mid_layer=links.Linear(hidden_units, hidden_units, wscale=w_scale),
            mid_rec_layer=links.LSTM(input_dim, rec_dim),
            output_layer=links.Linear(hidden_units, output_dim, wscale=w_scale))

    def __call__(self, x, test=False):
        h = self.input_layer(x)
        h = fun.relu(h)
        rec_h = self.mid_rec_layer(x)
        h = self.mid_layer(h)
        h = fun.relu(h)
        h = self.output_layer(fun.concat((h, rec_h)))
        h = self.output_layer(h)
        return h
