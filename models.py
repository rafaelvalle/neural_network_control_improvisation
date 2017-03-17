""" critic, discriminator and generator models from the adversarial literature
and from ganlucinations
"""

import numpy as np
from lasagne.utils import floatX
try:
    from lasagne.layers.dnn import batch_norm_dnn as batch_norm
except ImportError:
    from lasagne.layers import batch_norm
import pdb


def tanh_temperature(x, temperature=1):
    from lasagne.nonlinearities import tanh
    return tanh(x * temperature)


def BatchNorm(layer, include=True, mean=0.0, std=0):
    if include:
        if std == 0:
            return batch_norm(layer)
        return batch_norm(
            layer,
            mean=floatX(np.zeros(layer.output_shape[1])+mean),
            inv_std=floatX(np.random.normal(0.0, std, layer.output_shape[1])))
    else:
        return layer


def build_generator(input_var, noise_size, cond_var=None, n_conds=0, arch=0,
                    with_BatchNorm=True):
    from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer, concat
    from lasagne.layers import Upscale2DLayer, Conv2DLayer
    from lasagne.layers import TransposedConv2DLayer as Deconv2DLayer
    from lasagne.nonlinearities import LeakyRectify, rectify
    from lasagne.init import GlorotUniform  #  Normal Orthogonal

    # non_lin = LeakyRectify(0.01)
    non_lin = rectify
    # init = Orthogonal(np.sqrt(2/(1+0.01**2)))
    # init = Normal(0.02, 0.0)
    init = GlorotUniform()

    layer = InputLayer(shape=(None, noise_size), input_var=input_var)
    if cond_var is not None:
        layer = BatchNorm(DenseLayer(
            layer, noise_size, nonlinearity=non_lin), with_BatchNorm)
        layer = concat([
            layer, InputLayer(shape=(None, n_conds), input_var=cond_var)])
    if arch == 'dcgan':
        # DCGAN
        layer = BatchNorm(DenseLayer(
            layer, 1024*4*4, W=init, b=None, nonlinearity=non_lin))
        layer = ReshapeLayer(layer, ([0], 1024, 4, 4))
        layer = BatchNorm(Deconv2DLayer(
            layer, 512, 5, stride=2, crop=(2, 2), W=init, b=None,
            output_size=8, nonlinearity=non_lin), with_BatchNorm)
        layer = BatchNorm(Deconv2DLayer(
            layer, 256, 5, stride=2, crop=(2, 2), W=init, b=None,
            output_size=16, nonlinearity=non_lin), with_BatchNorm)
        layer = BatchNorm(Deconv2DLayer(
            layer, 128, 5, stride=2, crop=(2, 2), W=init, b=None,
            output_size=32, nonlinearity=non_lin), with_BatchNorm)
        layer = BatchNorm(Deconv2DLayer(
            layer, 64, 5, stride=2, crop=(2, 2), W=init, b=None,
            output_size=64, nonlinearity=non_lin), with_BatchNorm)
        layer = Deconv2DLayer(
            layer, 1, 5, stride=2, crop=(2, 2), W=init, b=None,
            output_size=128, nonlinearity=tanh_temperature)
    elif arch == 'mnist':
        # Jan Schluechter MNIST generator
        # fully-connected layers
        layer = BatchNorm(DenseLayer(
            layer, 1024, W=init, b=None), with_BatchNorm)
        layer = BatchNorm(DenseLayer(
            layer, 1024, W=init, b=None), with_BatchNorm)
        # project and reshape
        layer = BatchNorm(DenseLayer(
            layer, 128*34*34, W=init, b=None), with_BatchNorm)
        layer = ReshapeLayer(layer, ([0], 128, 34, 34))
        # two fractional-stride convolutions
        layer = BatchNorm(Deconv2DLayer(
            layer, 256, 5, stride=2, crop='same', W=init, b=None,
            nonlinearity=non_lin), with_BatchNorm)
        layer = Deconv2DLayer(
            layer, 1, 6, stride=2, crop='full', W=init, b=None,
            nonlinearity=tanh_temperature)
    elif arch == 2:
        # non-overlapping transposed convolutions
        # fully-connected layers
        layer = BatchNorm(DenseLayer(
            layer, 1024, W=init, b=None), with_BatchNorm)
        layer = BatchNorm(DenseLayer(
            layer, 1024, W=init, b=None), with_BatchNorm)
        # project and reshape
        layer = BatchNorm(DenseLayer(layer, 256*36*36), with_BatchNorm)
        layer = ReshapeLayer(layer, ([0], 256, 36, 36))
        # two fractional-stride convolutions
        layer = BatchNorm(Deconv2DLayer(
            layer, 256, 4, stride=2, crop='full', b=None, nonlinearity=non_lin),
            with_BatchNorm)
        layer = Deconv2DLayer(
            layer, 1, 8, stride=2, crop='full', b=None,
            nonlinearity=tanh_temperature)
    elif arch == 3:
        # resize-convolution, more full layer weights less convolutions
        # fully-connected layers
        layer = BatchNorm(DenseLayer(
            layer, 1024, W=init, b=None), with_BatchNorm)
        layer = BatchNorm(DenseLayer(
            layer, 1024, W=init, b=None), with_BatchNorm)
        # project and reshape
        layer = BatchNorm(DenseLayer(layer, 32*68*68), with_BatchNorm)
        layer = ReshapeLayer(layer, ([0], 32, 68, 68))
        # resize-convolutions
        layer = BatchNorm(Conv2DLayer(
            layer, 256, 3, stride=1, pad='valid'), with_BatchNorm)
        layer = Upscale2DLayer(layer, (2, 2))
        layer = Conv2DLayer(
            layer, 1, 5, stride=1, pad='valid', nonlinearity=tanh_temperature)
    elif arch == 4:
        # resize-convolution, less full layer weights more convolutions
        # fully-connected layers
        layer = BatchNorm(DenseLayer(
            layer, 1024, W=init, b=None), with_BatchNorm)
        layer = BatchNorm(DenseLayer(
            layer, 1024, W=init, b=None), with_BatchNorm)
        # project and reshape
        layer = BatchNorm(DenseLayer(layer, 128*18*18), with_BatchNorm)
        layer = ReshapeLayer(layer, ([0], 128, 18, 18))
        # resize-convolutions
        layer = Upscale2DLayer(layer, (2, 2), mode='bilinear')
        layer = BatchNorm(Conv2DLayer(
            layer, 256, 3, stride=1, pad='valid', nonlinearity=non_lin),
            with_BatchNorm)
        layer = Upscale2DLayer(layer, (2, 2), mode='bilinear')
        layer = BatchNorm(Conv2DLayer(
            layer, 256, 3, stride=1, pad='valid', nonlinearity=non_lin),
            with_BatchNorm)
        layer = Upscale2DLayer(layer, (2, 2), mode='bilinear')
        layer = Conv2DLayer(
            layer, 1, 5, stride=1, pad='valid',
            nonlinearity=tanh_temperature)
    elif arch == 'crepe_up':
        # CREPE transposed with upscaling
        # fully-connected layers
        layer = BatchNorm(DenseLayer(
            layer, 1024, W=init, b=None), with_BatchNorm)
        layer = BatchNorm(DenseLayer(
            layer, 1024, W=init, b=None), with_BatchNorm)
        # project and reshape
        layer = BatchNorm(DenseLayer(layer, 2**15*1*3), with_BatchNorm)
        layer = ReshapeLayer(layer, ([0], 2**15, 1, 3))
        # temporal convolutions
        layer = BatchNorm(Deconv2DLayer(
            layer, 256, (1, 3), stride=1, crop=0, W=init, b=None,
            nonlinearity=non_lin), with_BatchNorm)
        layer = BatchNorm(Deconv2DLayer(
            layer, 256, (1, 3), stride=1, crop=0, W=init, b=None,
            nonlinearity=non_lin), with_BatchNorm)
        layer = BatchNorm(Deconv2DLayer(
            layer, 256, (1, 3), stride=1, crop=0, W=init, b=None,
            nonlinearity=non_lin), with_BatchNorm)
        layer = BatchNorm(Deconv2DLayer(
            layer, 256, (1, 3), stride=1, crop=0, W=init, b=None,
            nonlinearity=non_lin), with_BatchNorm)
        layer = Upscale2DLayer(layer, (1, 3), mode='repeat')
        layer = BatchNorm(Deconv2DLayer(
            layer, 512, (1, 9), stride=1, crop=0, W=init, b=None,
            nonlinearity=non_lin), with_BatchNorm)
        layer = Upscale2DLayer(layer, (1, 3), mode='repeat')
        layer = Deconv2DLayer(
            layer, 1, (128, 6), stride=1, crop=0, W=init, b=None,
            nonlinearity=tanh_temperature)
    elif arch == 'crepe_noup_a':
        # CREPE transposed no upscaling
        # fully-connected layer
        layer = BatchNorm(DenseLayer(
            layer, 1024, W=init, b=None), with_BatchNorm)
        # project and reshape
        layer = BatchNorm(DenseLayer(
            layer, 1024*1*3, W=init, b=None), with_BatchNorm)
        layer = ReshapeLayer(layer, ([0], 1024, 1, 3))
        # temporal convolutions
        layer = BatchNorm(Deconv2DLayer(
            layer, 256, (1, 3), stride=1, crop=0, W=init, b=None,
            nonlinearity=non_lin), with_BatchNorm)
        layer = BatchNorm(Deconv2DLayer(
            layer, 256, (1, 3), stride=1, crop=0, W=init, b=None,
            nonlinearity=non_lin), with_BatchNorm)
        layer = BatchNorm(Deconv2DLayer(
            layer, 256, (1, 3), stride=1, crop=0, W=init, b=None,
            nonlinearity=non_lin), with_BatchNorm)
        layer = BatchNorm(Deconv2DLayer(
            layer, 256, (1, 3), stride=1, crop=0, W=init, b=None,
            nonlinearity=non_lin), with_BatchNorm)
        layer = BatchNorm(Deconv2DLayer(
            layer, 256, (1, 3), stride=1, crop=0, W=init, b=None,
            nonlinearity=non_lin), with_BatchNorm)
        layer = BatchNorm(Deconv2DLayer(
            layer, 512, (1, 7), stride=1, crop=0, W=init, b=None,
            nonlinearity=non_lin), with_BatchNorm)
        layer = BatchNorm(Deconv2DLayer(
            layer, 1024, (128, 7), stride=3, crop=0, W=init, b=None,
            nonlinearity=non_lin), with_BatchNorm)
        layer = Deconv2DLayer(
            layer, 1, (1, 8), stride=1, crop=0, W=init, b=None,
            nonlinearity=tanh_temperature)
    elif arch == 'crepe_noup_b':
        # CREPE transposed no upscaling
        # fully-connected layer
        layer = BatchNorm(DenseLayer(layer, 1024))
        # project and reshape
        layer = BatchNorm(DenseLayer(layer, 1024*1*3))
        layer = ReshapeLayer(layer, ([0], 1024, 1, 3))
        # temporal convolutions
        layer = BatchNorm(Deconv2DLayer(
            layer, 256, (1, 3), stride=1, crop=0,
            nonlinearity=non_lin))
        layer = BatchNorm(Deconv2DLayer(
            layer, 256, (1, 3), stride=1, crop=0, nonlinearity=non_lin))
        layer = BatchNorm(Deconv2DLayer(
            layer, 256, (1, 3), stride=1, crop=0, nonlinearity=non_lin))
        layer = BatchNorm(Deconv2DLayer(
            layer, 256, (1, 3), stride=1, crop=0, nonlinearity=non_lin))
        layer = BatchNorm(Deconv2DLayer(
            layer, 256, (1, 3), stride=3, crop=0, nonlinearity=non_lin))
        layer = Deconv2DLayer(
            layer, 512, (1, 9), stride=1, crop=0, nonlinearity=non_lin)
        layer = Deconv2DLayer(
            layer, 1, (128, 8), stride=3, crop=0, nonlinearity=tanh_temperature)
    else:
        return None

    print("Generator output:", layer.output_shape)
    return layer


def build_generator_lstm(params, gate_params, cell_params, arch=1):
    from lasagne.layers import InputLayer, DenseLayer, concat
    from lasagne.layers import DropoutLayer
    from lasagne.layers.recurrent import LSTMLayer
    from lasagne.init import Constant, HeNormal
    if arch == 1:
        # input layers
        l_in = InputLayer(
            shape=params['input_shape'], input_var=params['input_var'],
            name='g_in')
        l_noise = InputLayer(
            shape=params['noise_shape'], input_var=params['noise_var'],
            name='g_noise')
        l_cond = InputLayer(
            shape=params['cond_shape'], input_var=params['cond_var'],
            name='g_cond')
        l_mask = InputLayer(
            shape=params['mask_shape'], input_var=params['mask_var'],
            name='g_mask')

        # recurrent layers for bidirectional network
        l_forward_data = LSTMLayer(
            l_in, params['n_units'][0], mask_input=l_mask,
            ingate=gate_params, forgetgate=gate_params,
            cell=cell_params, outgate=gate_params,
            learn_init=True, grad_clipping=params['grad_clip'],
            only_return_final=False,
            nonlinearity=params['non_linearities'][0])
        l_forward_noise = LSTMLayer(
            l_noise, params['n_units'][0], mask_input=l_mask,
            ingate=gate_params, forgetgate=gate_params,
            cell=cell_params, outgate=gate_params,
            learn_init=True, grad_clipping=params['grad_clip'],
            only_return_final=False,
            nonlinearity=params['non_linearities'][1])

        l_backward_data = LSTMLayer(
            l_in, params['n_units'][0], mask_input=l_mask,
            ingate=gate_params, forgetgate=gate_params,
            cell=cell_params, outgate=gate_params,
            learn_init=True, grad_clipping=params['grad_clip'],
            only_return_final=False, backwards=True,
            nonlinearity=params['non_linearities'][0])
        l_backward_noise = LSTMLayer(
            l_noise, params['n_units'][0], mask_input=l_mask,
            ingate=gate_params, forgetgate=gate_params,
            cell=cell_params, outgate=gate_params,
            learn_init=True, grad_clipping=params['grad_clip'],
            only_return_final=False, backwards=True,
            nonlinearity=params['non_linearities'][1])

        # concatenate output of forward and backward layers
        l_lstm_concat = concat(
            [l_forward_data, l_forward_noise, l_backward_data,
             l_backward_noise], axis=2)

        # dense layer on output of data and noise lstms, w/dropout
        l_lstm_dense = DenseLayer(
            DropoutLayer(l_lstm_concat, p=0.5),
            num_units=params['n_units'][1], num_leading_axes=2,
            W=HeNormal(gain='relu'), b=Constant(0.1),
            nonlinearity=params['non_linearities'][2])

        # batch norm for lstm dense
        # l_lstm_dense = lasagne.layer.BatchNorm(l_lstm_dense)

        # concatenate dense layer of lstsm with condition
        l_lstm_cond_concat = concat(
            [l_lstm_dense, l_cond], axis=2)

        # dense layer with dense layer lstm and condition, w/dropout
        l_out = DenseLayer(
            DropoutLayer(l_lstm_cond_concat, p=0.5),
            num_units=params['n_units'][2],
            num_leading_axes=2,
            W=HeNormal(gain=1.0), b=Constant(0.1),
            nonlinearity=params['non_linearities'][3])
    elif arch == 2:
        raise Exception("arch 2 not implemented")
    elif arch == 3:
        raise Exception("arch 2 not implemented")

    class Generator:
        def __init__(self, l_in, l_noise, l_cond, l_mask, l_out):
            self.l_in = l_in
            self.l_noise = l_noise
            self.l_cond = l_cond
            self.l_mask = l_mask
            self.l_out = l_out

    return Generator(l_in, l_noise, l_cond, l_mask, l_out)


def build_discriminator_lstm(params, gate_params, cell_params):
    from lasagne.layers import InputLayer, DenseLayer, concat
    from lasagne.layers.recurrent import LSTMLayer
    from lasagne.regularization import l2, regularize_layer_params
    # from layers import MinibatchLayer
    # input layers
    l_in = InputLayer(
        shape=params['input_shape'], name='d_in')
    l_mask = InputLayer(
        shape=params['mask_shape'], name='d_mask')

    # recurrent layers for bidirectional network
    l_forward = LSTMLayer(
        l_in, params['n_units'], grad_clipping=params['grad_clip'],
        ingate=gate_params, forgetgate=gate_params,
        cell=cell_params, outgate=gate_params,
        nonlinearity=params['non_linearities'][0], only_return_final=True,
        mask_input=l_mask)
    l_backward = LSTMLayer(
        l_in, params['n_units'], grad_clipping=params['grad_clip'],
        ingate=gate_params, forgetgate=gate_params,
        cell=cell_params, outgate=gate_params,
        nonlinearity=params['non_linearities'][1], only_return_final=True,
        mask_input=l_mask, backwards=True)

    # concatenate output of forward and backward layers
    l_concat = concat([l_forward, l_backward], axis=1)

    # minibatch layer on forward and backward layers
    # l_minibatch = MinibatchLayer(l_concat, num_kernels=100)

    # output layer
    l_out = DenseLayer(
        l_concat, num_units=params['n_output_units'],
        nonlinearity=params['non_linearities'][2])

    regularization = regularize_layer_params(
        l_out, l2) * params['regularization']

    class Discriminator:
        def __init__(self, l_in, l_mask, l_out):
            self.l_in = l_in
            self.l_mask = l_mask
            self.l_out = l_out
            self.regularization = regularization

    return Discriminator(l_in, l_mask, l_out)


def build_critic(input_var=None, cond_var=None, n_conds=0, arch=0,
                 with_BatchNorm=True):
    from lasagne.layers import (
        InputLayer, Conv2DLayer, DenseLayer, MaxPool2DLayer, concat,
        dropout, flatten)
    from lasagne.nonlinearities import rectify, LeakyRectify
    from lasagne.init import GlorotUniform  # Normal
    lrelu = LeakyRectify(0.2)
    layer = InputLayer(
        shape=(None, 1, 128, 128), input_var=input_var, name='d_in_data')
    # init = Normal(0.02, 0.0)
    init = GlorotUniform()

    if cond_var:
        # class: from data or from generator input
        layer_cond = InputLayer(
            shape=(None, n_conds), input_var=cond_var, name='d_in_condition')
        layer_cond = BatchNorm(DenseLayer(
            layer_cond, 1024, W=init, b=None, nonlinearity=lrelu),
            with_BatchNorm)
    if arch == 'dcgan':
        # DCGAN inspired
        layer = BatchNorm(Conv2DLayer(
            layer, 32, 4, stride=2, pad=1, W=init, b=None, nonlinearity=lrelu),
            with_BatchNorm)
        layer = BatchNorm(Conv2DLayer(
            layer, 64, 4, stride=2, pad=1, W=init, b=None, nonlinearity=lrelu),
            with_BatchNorm)
        layer = BatchNorm(Conv2DLayer(
            layer, 128, 4, stride=2, pad=1, W=init, b=None, nonlinearity=lrelu),
            with_BatchNorm)
        layer = BatchNorm(Conv2DLayer(
            layer, 256, 4, stride=2, pad=1, W=init, b=None, nonlinearity=lrelu),
            with_BatchNorm)
        layer = BatchNorm(Conv2DLayer(
            layer, 512, 4, stride=2, pad=1, W=init, b=None, nonlinearity=lrelu),
            with_BatchNorm)
        # fully-connected layer
        layer = BatchNorm(DenseLayer(
            layer, 1024, W=init, b=None, nonlinearity=lrelu), with_BatchNorm)
    elif arch == 'mnist':
        # Jan Schluechter's MNIST discriminator
        # two convolutions
        layer = BatchNorm(Conv2DLayer(
            layer, 64, 5, stride=2, pad='same', W=init, b=None,
            nonlinearity=lrelu), with_BatchNorm)
        layer = BatchNorm(Conv2DLayer(
            layer, 128, 5, stride=2, pad='same', W=init, b=None,
            nonlinearity=lrelu), with_BatchNorm)
        # fully-connected layer
        layer = BatchNorm(DenseLayer(
            layer, 1024, W=init, b=None, nonlinearity=lrelu), with_BatchNorm)
    elif arch == 'crepe':
        # CREPE
        # form words from sequence of characters
        layer = BatchNorm(Conv2DLayer(
            layer, 1024, (128, 7), W=init, b=None, nonlinearity=lrelu),
            with_BatchNorm)
        layer = MaxPool2DLayer(layer, (1, 3))
        # temporal convolution, 7-gram
        layer = BatchNorm(Conv2DLayer(
            layer, 512, (1, 7), W=init, b=None, nonlinearity=lrelu),
            with_BatchNorm)
        layer = MaxPool2DLayer(layer, (1, 3))
        # temporal convolution, 3-gram
        layer = BatchNorm(Conv2DLayer(
            layer, 256, (1, 3), W=init, b=None, nonlinearity=lrelu),
            with_BatchNorm)
        layer = BatchNorm(Conv2DLayer(
            layer, 256, (1, 3), W=init, b=None, nonlinearity=lrelu),
            with_BatchNorm)
        layer = BatchNorm(Conv2DLayer(
            layer, 256, (1, 3), W=init, b=None, nonlinearity=lrelu),
            with_BatchNorm)
        layer = BatchNorm(Conv2DLayer(
            layer, 256, (1, 3), W=init, b=None, nonlinearity=lrelu),
            with_BatchNorm)
        layer = flatten(layer)
        # fully-connected layers
        layer = dropout(DenseLayer(
            layer, 1024, W=init, b=None, nonlinearity=rectify))
        layer = dropout(DenseLayer(
            layer, 1024, W=init, b=None, nonlinearity=rectify))
    else:
        raise Exception("Model architecture {} is not supported".format(arch))
        # output layer (linear and without bias)
    if cond_var is not None:
        layer = concat([layer, layer_cond])
    layer = DenseLayer(layer, 1, nonlinearity=None, b=None)
    print("Critic output:", layer.output_shape)
    return layer
