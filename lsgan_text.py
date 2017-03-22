#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example employing Lasagne for digit generation using the MNIST dataset and
Least Squares Generative Adversarial Networks
(LSGANs, see https://arxiv.org/abs/1611.04076 for the paper).

It is based on a WGAN example:
https://gist.github.com/f0k/f3190ebba6c53887d598d03119ca2066
This, in turn, is based on a DCGAN example:
https://gist.github.com/f0k/738fa2eedd9666b78404ed1751336f56
This, in turn, is based on the MNIST example in Lasagne:
https://lasagne.readthedocs.io/en/latest/user/tutorial.html

Jan Schlüter, 2017-03-07
"""

from __future__ import print_function
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne

import cPickle as pkl
from tqdm import tqdm

from data_processing import load_text_data, encode_labels
from text_utils import textEncoder
import pdb

def pad(cur_data, alphabet_size, length, padding):
    if padding is None:
        # ignore examples shorter than length
       return cur_data
    elif padding == 'zero':
        # zero pad data if shorter than length
        z = np.zeros((alphabet_size, length))
        z[:, :cur_data.shape[1]] = cur_data
    elif padding == 'noise':
        z = np.zeros((alphabet_size, length))
        z[:, :cur_data.shape[1]] = cur_data
        z[:, cur_data.shape[1]:] = np.random.normal(
            0, 0.0001,
            (cur_data.shape[0], length - cur_data.shape[1]))
    elif padding == 'repeat':
        z = np.zeros((alphabet_size, length))
        z[:, :cur_data.shape[1]] = cur_data
        # change to vector operation
        for k in range(1, length - cur_data.shape[1]-1):
            z[:, cur_data.shape[1] + k] = cur_data[
                :, (k-1) % cur_data.shape[1]]
    else:
        raise Exception(
            "Padding {} not supported".format(padding))
    return z


def iterate_minibatches_text(inputs, labels, batch_size, encoder, shuffle=True,
                             forever=True, length=128, alphabet_size=128,
                             padding='repeat'):
    from text_utils import binarizeText
    if shuffle:
        indices = np.arange(len(inputs))
    while True:
        if shuffle:
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)

            data = []
            # select random slice from each piano roll
            for i in excerpt:
                cur_data = binarizeText(
                    inputs[i], encoder, lower_case=True, remove_html=False)
                if cur_data.shape[1] < length:
                    cur_data = pad(cur_data, alphabet_size, length, padding)
                elif cur_data.shape[1] > length:
                    # slice data if larger than length
                    rand_start = np.random.randint(
                        0, cur_data.shape[1] - length)
                    cur_data = cur_data[:, rand_start:rand_start+length]
                data.append(cur_data)
            # scale to [-1, 1]
            data = np.array(data)
            data += data.min()
            data /= data.max()
            data = (data * 2) - 1
            yield lasagne.utils.floatX(data), labels[excerpt]

        if not forever:
            break

# ##################### Build the neural network model #######################
# We create two models: The generator and the critic network.
# The models are the same as in the Lasagne DCGAN example, except that the
# discriminator is now a critic with linear output instead of sigmoid output.

def build_generator(input_var=None, batch_size=None, n_timesteps=128, alphabet_size=128):
    from lasagne.layers import InputLayer, DenseLayer, LSTMLayer
    from lasagne.layers import TransposedConv2DLayer as Deconv2DLayer
    from lasagne.layers import ExpressionLayer, NonlinearityLayer
    from lasagne.layers import ReshapeLayer, DimshuffleLayer, Upscale2DLayer, concat
    try:
        from lasagne.layers.dnn import batch_norm_dnn as batch_norm
    except ImportError:
        from lasagne.layers import batch_norm

    from lasagne.nonlinearities import sigmoid, tanh, softmax
    """
    layer = InputLayer(shape=(batch_size, 100), input_var=input_var)
    print("MNIST generator")
    layer = batch_norm(DenseLayer(layer, 1024))
    layer = batch_norm(DenseLayer(layer, 1024*8*8))
    layer = ReshapeLayer(layer, ([0], 1024, 8, 8))
    layer = batch_norm(Deconv2DLayer(
        layer, 128, 5, stride=2, crop='same', output_size=16))
    layer = batch_norm(Deconv2DLayer(
        layer, 128, 5, stride=2, crop='same', output_size=32))
    layer = batch_norm(Deconv2DLayer(
        layer, 128, 5, stride=2, crop='same', output_size=64))
    layer = batch_norm(Deconv2DLayer(
        layer, 1, 5, stride=2, crop='same', output_size=128,
        nonlinearity=tanh))

    # Crepe
    print("Crepe generator")
    layer = batch_norm(DenseLayer(layer, 1024))
    layer = batch_norm(DenseLayer(layer, 1024*13))
    layer = ReshapeLayer(layer, ([0], 1024, 1, 13))
    layer = batch_norm(Deconv2DLayer(
        layer, 512, (1, 4), stride=2, crop=0))
    layer = batch_norm(Deconv2DLayer(
        layer, 1024, (1, 5), stride=2, crop=0))
    layer = batch_norm(Deconv2DLayer(
        layer, 2048, (1, 5), stride=2, crop=0))
    layer = Deconv2DLayer(
        layer, 1, (128, 8), stride=1, crop=0, nonlinearity=tanh)
    """
    # LSTM
    # input layers
    layer = InputLayer(shape=(batch_size, n_timesteps, 100), input_var=input_var)
    # recurrent layers for bidirectional network
    l_forward_noise = LSTMLayer(
        layer, 64, learn_init=True, grad_clipping=None, only_return_final=False)
    l_backward_noise = LSTMLayer(
        layer, 64, learn_init=True, grad_clipping=None, only_return_final=False,
        backwards=True)
    layer = concat(
        [l_forward_noise, l_backward_noise], axis=2)
    pdb.set_trace()
    layer = DenseLayer(layer, 1024, num_leading_axes=2)
    layer = DenseLayer(layer, alphabet_size, num_leading_axes=2)
    layer = ReshapeLayer(layer, (batch_size*n_timesteps, -1))
    layer = NonlinearityLayer(layer, softmax)
    layer = ReshapeLayer(layer, (batch_size, n_timesteps, -1))
    layer = DimshuffleLayer(layer, (0, 'x', 2, 1))
    layer = ExpressionLayer(layer, lambda X: X*2 - 1)

    print("Generator output:", layer.output_shape)
    return layer


def build_critic(input_var=None):
    from lasagne.layers import (InputLayer, Conv2DLayer, ReshapeLayer,
                                DenseLayer, MaxPool2DLayer, dropout)
    try:
        from lasagne.layers.dnn import batch_norm_dnn as batch_norm
    except ImportError:
        from lasagne.layers import batch_norm
    from lasagne.nonlinearities import LeakyRectify, rectify
    lrelu = LeakyRectify(0.2)
    layer = InputLayer(
        shape=(None, 1, 128, 128), input_var=input_var, name='d_in_data')

    print("MNIST critic")
    # convolution layers
    layer = batch_norm(Conv2DLayer(
        layer, 128, 5, stride=2, pad='same', nonlinearity=lrelu))
    layer = batch_norm(Conv2DLayer(
        layer, 128, 5, stride=2, pad='same', nonlinearity=lrelu))
    layer = batch_norm(Conv2DLayer(
        layer, 128, 5, stride=2, pad='same', nonlinearity=lrelu))
    """
    print("naive CREPE critic")
    # words from sequences with 7 characters
    # each filter learns a word representation of shape M x 1
    layer = Conv2DLayer(
        layer, 128, (128, 7), nonlinearity=lrelu)
    layer = MaxPool2DLayer(layer, (1, 3))
    # temporal convolution, 7-gram
    layer = Conv2DLayer(
        layer, 128, (1, 7), nonlinearity=lrelu)
    layer = MaxPool2DLayer(layer, (1, 3))
    # temporal convolution, 3-gram
    layer = Conv2DLayer(
        layer, 128, (1, 3), nonlinearity=lrelu)
    layer = Conv2DLayer(
        layer, 128, (1, 3), nonlinearity=lrelu)
    layer = Conv2DLayer(
        layer, 128, (1, 3), nonlinearity=lrelu)
    layer = Conv2DLayer(
        layer, 128, (1, 3), nonlinearity=lrelu)
    # fully-connected layers
    layer = DenseLayer(layer, 1024, nonlinearity=rectify)
    layer = DenseLayer(layer, 1024, nonlinearity=rectify)
    """
    layer = DenseLayer(layer, 1, nonlinearity=lrelu)
    print("critic output:", layer.output_shape)
    return layer


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batch_size, shuffle=False,
                        forever=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
    while True:
        if shuffle:
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield inputs[excerpt], targets[excerpt]
        if not forever:
            break


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(num_epochs=100, epochsize=100, batch_size=32, initial_deta=1e-4,
         initial_geta=1e-2):
    # Load the data according to datatype
    datapaths = (
        '/Users/rafaelvalle/Desktop/datasets/TEXT/ag_news_csv/train.csv', )
        #'/media/steampunkhd/rafaelvalle/datasets/TEXT/ag_news_csv/test.csv')

    data_cols = (2, 1)
    label_cols = (0, 0)
    n_pieces = 0  # 0 is equal to all pieces, unbalanced dataset
    as_dict = False
    n_timesteps = 128
    padding = 'repeat'
    inputs, labels = load_text_data(
        datapaths, data_cols, label_cols, n_pieces, as_dict,
        patch_size=n_timesteps)

    data_size = 5000
    inputs = inputs[:data_size]
    labels = labels[:data_size]
    iterator = iterate_minibatches_text
    # alphabet = list("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}")
    alphabet = [chr(x) for x in range(128)]
    encoder = textEncoder(alphabet)
    alphabet_size = len(encoder.alphabet)
    pkl.dump(encoder, open("encdec.pkl", "wb"))

    # encode labels
    labels = encode_labels(labels, one_hot=True).astype(np.float32)
    print("Dataset shape {}".format(inputs.shape))

    # Prepare Theano variables for inputs and targets
    # noise_var = T.fmatrix('noise')
    noise_var = T.ftensor3('noise')
    input_var = T.tensor4('inputs')


    # Create neural network model
    print("Building model and compiling functions...")
    generator = build_generator(noise_var, batch_size, n_timesteps, alphabet_size)
    critic = build_critic(input_var)
    print("Generator #params {}".format(
        lasagne.layers.count_params(generator)))
    print("Critic #params {}".format(
        lasagne.layers.count_params(critic)))

    # Create expression for passing real data through the critic
    real_out = lasagne.layers.get_output(critic)
    # Create expression for passing fake data through the critic
    fake_out = lasagne.layers.get_output(
        critic, lasagne.layers.get_output(generator))

    # Create loss expressions to be minimized
    # a, b, c = -1, 1, 0  # Equation (8) in the paper
    a, b, c = 0, 1, 1  # Equation (9) in the paper
    generator_loss = lasagne.objectives.squared_error(fake_out, c).mean()
    critic_loss = (lasagne.objectives.squared_error(real_out, b).mean() +
                   lasagne.objectives.squared_error(fake_out, a).mean())

    # Create update expressions for training
    generator_params = lasagne.layers.get_all_params(generator, trainable=True)
    critic_params = lasagne.layers.get_all_params(critic, trainable=True)
    deta = theano.shared(lasagne.utils.floatX(initial_deta))
    geta = theano.shared(lasagne.utils.floatX(initial_geta))
    generator_updates = lasagne.updates.rmsprop(
            generator_loss, generator_params, learning_rate=geta)
    critic_updates = lasagne.updates.rmsprop(
            critic_loss, critic_params, learning_rate=deta)

    # Instantiate a symbolic noise generator to use for training
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    srng = RandomStreams(seed=np.random.randint(2147462579, size=6))
    # noise = srng.normal((batch_size, 100))
    noise = srng.uniform((batch_size, 128, 100))

    # Compile functions performing a training step on a mini-batch (according
    # to the updates dictionary) and returning the corresponding score:
    generator_train_fn = theano.function([], generator_loss,
                                         givens={noise_var: noise},
                                         updates=generator_updates)
    critic_train_fn = theano.function([input_var], critic_loss,
                                      givens={noise_var: noise},
                                      updates=critic_updates)

    # Compile another function generating some data
    gen_fn = theano.function([noise_var],
                             lasagne.layers.get_output(generator,
                                                       deterministic=True))

    # compile function for computing gradients
    # grads = theano.grad(generator_loss, generator_params)
    # gen_grad_fn = theano.function([noise_var], grads)

    # We create an infinite supply of batches (as an iterable generator):
    print("Loading data...")
    batches = iterator(
        inputs, labels, batch_size, encoder, shuffle=True, length=n_timesteps,
        forever=True, alphabet_size=len(encoder.alphabet), padding=padding)

    # create fixed-noise
    fixed_noise = lasagne.utils.floatX(np.random.rand(32, 128, 100))
    # fixed_noise = lasagne.utils.floatX(np.random.rand(32, 100))

    # We iterate over epochs:
    generator_updates = 0
    epoch_critic_losses = []
    epoch_generator_losses = []
    # Finally, launch the training loop.
    print("Starting training...")
    for epoch in range(num_epochs):
        start_time = time.time()

        # In each epoch, we do `epochsize` generator and critic updates.
        critic_losses = []
        generator_losses = []
        for _ in tqdm(range(epochsize)):
            inputs, _ = next(batches)
            # reshape batch to proper dimensions
            inputs = inputs.reshape(
                inputs.shape[0], 1, inputs.shape[1], inputs.shape[2])
            critic_losses.append(critic_train_fn(inputs))
            generator_losses.append(generator_train_fn())
        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  generator loss: {}".format(np.mean(generator_losses)))
        print("  critic loss:    {}".format(np.mean(critic_losses)))

        # And finally, we plot some generated data
        samples = gen_fn(fixed_noise)

        plt.imsave('images/lsgan_text/lsgan_sample_{}.png'.format(epoch),
                   (samples.reshape(4, 8, len(encoder.alphabet), n_timesteps)
                           .transpose(0, 2, 1, 3)
                           .reshape(4*len(encoder.alphabet), 8*n_timesteps)),
                   origin='bottom',
                   cmap='gray')

        epoch_critic_losses.append(np.mean(critic_losses))
        epoch_generator_losses.append(np.mean(generator_losses))
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        axes[0].set_title('Loss(C)')
        axes[0].plot(epoch_critic_losses)
        axes[1].set_title('Loss(G)')
        axes[1].plot(epoch_generator_losses)
        fig.tight_layout()
        fig.savefig('images/lsgan_text/lsgan_epoch_{}'.format(epoch))
        plt.close('all')

        # After half the epochs, we start decaying the learn rate towards zero
        if epoch >= num_epochs // 2:
            progress = float(epoch) / num_epochs
            deta.set_value(lasagne.utils.floatX(initial_deta*2*(1 - progress)))
            geta.set_value(lasagne.utils.floatX(initial_geta*2*(1 - progress)))

    # Optionally, you could now dump the network weights to a file like this:
    np.savez('lsgan_text_gen.npz',
             *lasagne.layers.get_all_param_values(generator))
    np.savez('lsgan_text_crit.npz',
             *lasagne.layers.get_all_param_values(critic))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a LSGAN on MNIST using Lasagne.")
        print("Usage: %s [EPOCHS [EPOCHSIZE]]" % sys.argv[0])
        print()
        print("EPOCHS: number of training epochs to perform (default: 1000)")
        print("EPOCHSIZE: number of network updates per epoch (default: 100)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['num_epochs'] = int(sys.argv[1])
        if len(sys.argv) > 2:
            kwargs['epochsize'] = int(sys.argv[2])
        main(**kwargs)
