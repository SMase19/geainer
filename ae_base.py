import doctest

import chainer
import chainer.functions as f
import chainer.links as l
from chainer.functions.loss.vae import gaussian_kl_divergence
from utils.functions import sigmoid_cross_entropy, up_sampling_2d
from chainer import initializer, cuda


class AutoEncoderBase(chainer.Chain):

    def encode(self, x):
        pass

    def decode_bottleneck(self, z):
        pass

    def decode(self, z):
        return f.sigmoid(self.decode_bottleneck(z))

    def bottleneck(self, x):
        return self.decode_bottleneck(self.encode(x))

    def __call__(self, x):
        return self.decode(self.encode(x))

    def get_loss_func(self, *args, **kwargs):
        def lf(x):
            y = self.bottleneck(x)
            self.loss = sigmoid_cross_entropy(y, x)
            self.rec_loss = self.loss / y.data.shape[0]
            # self.rec_loss = F.mean_squared_error(F.sigmoid(y), x)
            return self.loss
        return lf


class SimpleAutoEncoder(AutoEncoderBase):

    def __init__(self, n_in, n_units):
        super(SimpleAutoEncoder, self).__init__()
        with self.init_scope():
            self.l1 = l.Linear(n_in, n_units)
            self.l2 = l.Linear(n_units, n_in)

    def encode(self, x):
        return f.relu(self.l1(x))

    def decode_bottleneck(self, z):
        return self.l2(z)


class SparseAutoEncoder(SimpleAutoEncoder):

    def get_loss_func(self, l1=0.01, l2=0.0, *args, **kwargs):
        from utils import functions

        def lf(x):
            # bottleneck of AutoEncoderBase
            y = self.bottleneck(x)
            self.loss = sigmoid_cross_entropy(y, x)
            self.rec_loss = self.loss / y.data.shape[0]
            self.loss += functions.l1_norm(y) * l1
            return self.loss
        return lf


class DeepAutoEncoder(AutoEncoderBase):

    def __init__(self, n_in, n_depth=1, n_units=None):
        super(DeepAutoEncoder, self).__init__()
        with self.init_scope():
            self.n_in = n_in
            self.n_units = n_units
            self.encode_label = "encode%d"
            self.decode_label = "decode%d"
            self.n_depth = 0
            self._init_layers(n_depth)

    def encode(self, x):
        for n in range(self.n_depth):
            x = f.relu(self[self.decode_label % n](x))
        return x

    def decode_bottleneck(self, z):
        d = self.n_depth
        for n in reversed(range(1, d)):
            z = f.relu(self[self.decode_label % n](z))
        return self[self.decode_label % 0](z)

    def _init_layers(self, n_depth):
        """
        >>> [p.data.shape for p in DeepAutoEncoder(784, n_depth=2, n_units=32).params()]
        [(64, 784), (64,), (784, 64), (784,), (32, 64), (32,), (64, 32), (64,)]
        >>> [p.data.shape for p in DeepAutoEncoder(784, n_depth=2).params()]
        [(392, 784), (392,), (784, 392), (784,), (196, 392), (196,), (392, 196), (392,)]
        """
        first = None
        if self.n_units:
            first = self.n_units * (2 ** (n_depth - 1))
        self.add_layer(first)
        for n in range(1, n_depth):
            self.add_layer()

    def encode_size(self):
        """
        >>> DeepAutoEncoder(3, n_units=7).encoded_size()
        7
        >>> DeepAutoEncoder(9, n_depth=2).encoded_size()
        2
        """
        if self.n_depth == 0:
            return self.n_in
        last = self.n_depth - 1
        encode_last = self[self.encode_label % last]
        return encode_last.b.data.size

    def add_layer(self, n_out = None):
        i = self.encode_size()
        o = i // 2 if n_out is None else n_out
        self.add_link(self.encode_label % self.n_depth, f.linear(i, o))
        self.add_link(self.decode_label % self.n_depth, f.linear(o, i))
        self.n_depth += 1


class ConvolutionalAutoEncoder(AutoEncoderBase):

    def __init__(self, n_in=784):
        self.n_in_square = int(n_in**0.5)
        p = 1
        q = 0
        super(ConvolutionalAutoEncoder, self).__init__()
        with self.init_scope():
            # encoder input (1, 28, 28)
            self.conv0=l.Convolution2D(1, 16, 3, pad=p), # (28, 28)  -> (14, 14)
            self.conv1=l.Convolution2D(16, 8, 3, pad=p), # (14, 14)  -> (7, 7)
            self.conv2=l.Convolution2D(8, 8, 3, pad=p),  # (7, 7)    -> (4, 4)
            # decoder
            self.conv3=l.Convolution2D(8, 8, 3, pad=p),  # (4, 4)    ->  (8, 8)
            self.conv4=l.Convolution2D(8, 8, 3, pad=p),  # (4, 4)    ->  (8, 8)
            self.conv5=l.Convolution2D(8, 16, 3, pad=q),  # (8, 8)    ->  (16)
            self.conv6=l.Convolution2D(16, 1, 3, pad=p),

        self.n_depth = 3
        self.label = "conv%d"

    def reshape_2d(self, x):
        return x.reshape(-1, 1, self.n_in_square, self.n_in_square)

    def encode(self, x):
        x.data = self.reshape_2d(x.data)
        for n in range(self.n_depth):
            conv = self[self.label % n]
            x = f.relu(conv(x))
            x = f.max_pooling_2d(x, 2)
        return x

    def decode_bottleneck(self, z):
        last = self.n_depth * 2
        for n in range(self.n_depth, last):
            conv = self[self.label % n]
            z = f.relu(conv(z))
            z = up_sampling_2d(z, 2)
        z = self[self.label % last](z)
        return z

import numpy as np


# Reference: https://jmetzen.github.io/2015-11-27/vae.html


class Xavier(initializer.Initializer):
    """
    Xavier initializaer
    Reference:
    * https://jmetzen.github.io/2015-11-27/vae.html
    * https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    """

    def __init__(self, fan_in, fan_out, constant=1, dtype=None):
        #self.fan_in = fan_in
        #self.fan_out = fan_out
        self.high = constant*np.sqrt(6.0/(fan_in + fan_out))
        self.low = -self.high
        super(Xavier, self).__init__(dtype)

    def __call__(self, array):
        xp = cuda.get_array_module(array)
        args = {'low': self.low, 'high': self.high, 'size': array.shape}
        if xp is not np:
            # Only CuPy supports dtype option
            if self.dtype == np.float32 or self.dtype == np.float16:
                # float16 is not supported in cuRAND
                args['dtype'] = np.float32
        array[...] = xp.random.uniform(**args)


class TybaltVAE(chainer.Chain):
    """Variational AutoEncoder"""

    def __init__(self, n_in, n_latent, n_h, act_func=f.tanh):
        super(TybaltVAE, self).__init__()
        self.act_func = act_func
        with self.init_scope():
            # encoder
            self.le1 = l.Linear(n_in, n_h, initialW=Xavier(n_in, n_h))
            self.bn1 = l.BatchNormalization(n_h, use_gamma=False, use_beta=False)
            self.le2_mean = l.Linear(n_h, n_latent, initialW=Xavier(n_h, n_latent))
            self.le2_log_var = l.Linear(n_h, n_latent, initialW=Xavier(n_h, n_latent))
            # decoder
            self.ld1 = l.Linear(n_latent, n_in, initialW=Xavier(n_latent, n_h))

    def __call__(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        if type(x) != chainer.variable.Variable:
            x = chainer.Variable(x)
        x.name = "x"
        # h1 = self.act_func(self.bn1(self.le1(x)))
        le1 = self.le1(x)
        bn1 = self.bn1(le1)
        h1 = self.act_func(bn1)
        h1.name = "enc_h1"
        mean = self.le2_mean(h1)
        mean.name = "z_mean"
        log_var = self.le2_log_var(h1)
        log_var.name = "z_log_var"
        return mean, log_var

    def decode(self, z, sigmoid=True):
        h1 = self.ld1(z)
        h1.name = "dec_h1"
        if sigmoid:
            return f.sigmoid(h1)
        else:
            return h1

    def get_loss_func(self, C=1.0, k=1):
        """Get loss function of VAE.
        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.
        Args:
            C (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """
        def lf(x):
            mu, ln_var = self.encode(x)
            batch_size = len(mu.data)
            # reconstruction loss
            rec_loss = 0
            for l in range(k):
                z = f.gaussian(mu, ln_var)
                z.name = "z"
                rec_loss += f.bernoulli_nll(x, self.decode(z, sigmoid=True)) \
                    / (k * batch_size)
            self.rec_loss = rec_loss
            self.rec_loss.name = "reconstruction error"
            self.latent_loss = C * gaussian_kl_divergence(mu, ln_var) / batch_size
            self.latent_loss.name = "latent loss"
            self.loss = self.rec_loss + self.latent_loss
            self.loss.name = "loss"
            return self.loss
        return lf


class VariationalAutoEncoder(AutoEncoderBase):
    """Variational AutoEncoder"""
    def __init__(self, n_in, n_latent, n_h):
        super(VariationalAutoEncoder, self).__init__(
            # encoder
            le1=l.Linear(n_in, n_h),
            le2_mu=l.Linear(n_h, n_latent),
            le2_ln_var=l.Linear(n_h, n_latent),
            # decoder
            ld1=l.Linear(n_latent, n_h),
            ld2=l.Linear(n_h, n_in),
        )
        self.loss = None
        self.n_latent = n_latent

    def __call__(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        h1 = f.tanh(self.le1(x))
        mu = self.le2_mu(h1)
        ln_var = self.le2_ln_var(h1)  # log(sigma**2)
        return mu, ln_var

    def decode(self, z, sigmoid=True):
        h1 = f.tanh(self.ld1(z))
        h2 = self.ld2(h1)
        if sigmoid:
            return f.sigmoid(h2)
        else:
            return h2

    def get_loss_func(self, C=1.0, k=1, train=True):
        """Get loss function of VAE.

        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.

        Args:
            C (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
            train (bool): If true loss_function is used for training.
        """
        def lf(x):
            mu, ln_var = self.encode(x)
            batchsize = len(mu.data)
            # reconstruction loss
            rec_loss = 0
            for l in range(k):
                z = f.gaussian(mu, ln_var)
                rec_loss += f.bernoulli_nll(x, self.decode(z, sigmoid=False)) \
                    / (k * batchsize)
            self.rec_loss = rec_loss
            self.loss = self.rec_loss + \
                C * gaussian_kl_divergence(mu, ln_var) / batchsize
            return self.loss
        return lf


if __name__ == '__main__':
    doctest.testmod()
