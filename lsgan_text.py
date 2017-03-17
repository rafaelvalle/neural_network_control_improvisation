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


def iterate_minibatches_text(inputs, labels, batchsize, encoder, shuffle=True,
                             forever=True, length=128, alphabet_size=128,
                             padding='repeat'):
    from text_utils import binarizeText
    if shuffle:
        indices = np.arange(len(inputs))
    while True:
        if shuffle:
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)

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

def build_generator(input_var=None):
    from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer
    try:
        from lasagne.layers import TransposedConv2DLayer as Deconv2DLayer
    except ImportError:
        raise ImportError("Your Lasagne is too old. Try the bleeding-edge "
                          "version: http://lasagne.readthedocs.io/en/latest/"
                          "user/installation.html#bleeding-edge-version")
    try:
        from lasagne.layers.dnn import batch_norm_dnn as batch_norm
    except ImportError:
        from lasagne.layers import batch_norm
    from lasagne.nonlinearities import sigmoid, tanh
    """
    # input: 100dim
    layer = InputLayer(shape=(None, 100), input_var=input_var)
    # fully-connected layer
    layer = batch_norm(DenseLayer(layer, 1024))
    # project and reshape
    layer = batch_norm(DenseLayer(layer, 128*7*7))
    layer = ReshapeLayer(layer, ([0], 128, 7, 7))
    # two fractional-stride convolutions
    layer = batch_norm(Deconv2DLayer(layer, 64, 5, stride=2, crop='same',
                                     output_size=14))
    layer = Deconv2DLayer(layer, 1, 5, stride=2, crop='same', output_size=28,
                          nonlinearity=sigmoid)
    """
    # fully-connected layers
    layer = InputLayer(shape=(None, 100), input_var=input_var)
    layer = batch_norm(DenseLayer(layer, 1024))
    layer = batch_norm(DenseLayer(layer, 1024))
    # project and reshape
    layer = batch_norm(DenseLayer(layer, 128*34*34))
    layer = ReshapeLayer(layer, ([0], 128, 34, 34))
    # two fractional-stride convolutions
    layer = batch_norm(Deconv2DLayer(
        layer, 256, 5, stride=2, crop='same'))
    layer = Deconv2DLayer(
        layer, 1, 6, stride=2, crop='full',
        nonlinearity=sigmoid)
    print ("Generator output:", layer.output_shape)
    return layer

def build_critic(input_var=None):
    from lasagne.layers import (InputLayer, Conv2DLayer, ReshapeLayer,
                                DenseLayer)
    try:
        from lasagne.layers.dnn import batch_norm_dnn as batch_norm
    except ImportError:
        from lasagne.layers import batch_norm
    from lasagne.nonlinearities import LeakyRectify
    lrelu = LeakyRectify(0.2)
    """
    # input: (None, 1, 28, 28)
    layer = InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
    # two convolutions
    layer = batch_norm(Conv2DLayer(layer, 64, 5, stride=2, pad='same',
                                   nonlinearity=lrelu))
    layer = batch_norm(Conv2DLayer(layer, 128, 5, stride=2, pad='same',
                                   nonlinearity=lrelu))
    # fully-connected layer
    layer = batch_norm(DenseLayer(layer, 1024, nonlinearity=lrelu))
    # output layer (linear)
    """
    layer = InputLayer(
        shape=(None, 1, 128, 128), input_var=input_var, name='d_in_data')
    # two convolutions
    layer = batch_norm(Conv2DLayer(
        layer, 64, 5, stride=2, pad='same',
        nonlinearity=lrelu))
    layer = batch_norm(Conv2DLayer(
        layer, 128, 5, stride=2, pad='same',
        nonlinearity=lrelu))
    # fully-connected layer
    layer = batch_norm(DenseLayer(
        layer, 1024, nonlinearity=lrelu))
    layer = DenseLayer(layer, 1, nonlinearity=lrelu)
    print ("critic output:", layer.output_shape)
    return layer


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False,
                        forever=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
    while True:
        if shuffle:
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]
        if not forever:
            break


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(num_epochs=1000, epochsize=100, batchsize=32, initial_eta=1e-4):
    # Load the data according to datatype
    datapaths = (
        '/Users/rafaelvalle/Desktop/datasets/TEXT/ag_news_csv/train.csv', )
        #'/media/steampunkhd/rafaelvalle/datasets/TEXT/ag_news_csv/test.csv')

    data_cols = (2, 1)
    label_cols = (0, 0)
    n_pieces = 0  # 0 is equal to all pieces, unbalanced dataset
    as_dict = False
    patch_size = 128
    padding = 'repeat'
    inputs, labels = load_text_data(
        datapaths, data_cols, label_cols, n_pieces, as_dict,
        patch_size=patch_size)

    data_size = 5000
    inputs = inputs[:data_size]
    labels = labels[:data_size]
    iterator = iterate_minibatches_text
    # alphabet = list("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}")
    alphabet = [chr(x) for x in range(128)]
    encoder = textEncoder(alphabet)
    pkl.dump(encoder, open("encdec.pkl", "wb"))

    # encode labels
    labels = encode_labels(labels, one_hot=True).astype(np.float32)
    print("Dataset shape {}".format(inputs.shape))

    # create fixed conditions and noise for output evaluation
    n_samples = 12 * 12
    fixed_noise = lasagne.utils.floatX(np.random.rand(n_samples, 100))


    # Prepare Theano variables for inputs and targets
    noise_var = T.matrix('noise')
    input_var = T.tensor4('inputs')

    # Create neural network model
    print("Building model and compiling functions...")
    generator = build_generator(noise_var)
    critic = build_critic(input_var)

    # Create expression for passing real data through the critic
    real_out = lasagne.layers.get_output(critic)
    # Create expression for passing fake data through the critic
    fake_out = lasagne.layers.get_output(critic,
            lasagne.layers.get_output(generator))

    # Create loss expressions to be minimized
    # a, b, c = -1, 1, 0  # Equation (8) in the paper
    a, b, c = 0, 1, 1  # Equation (9) in the paper
    generator_loss = lasagne.objectives.squared_error(fake_out, c).mean()
    critic_loss = (lasagne.objectives.squared_error(real_out, b).mean() +
                   lasagne.objectives.squared_error(fake_out, a).mean())

    # Create update expressions for training
    generator_params = lasagne.layers.get_all_params(generator, trainable=True)
    critic_params = lasagne.layers.get_all_params(critic, trainable=True)
    eta = theano.shared(lasagne.utils.floatX(initial_eta))
    generator_updates = lasagne.updates.rmsprop(
            generator_loss, generator_params, learning_rate=eta)
    critic_updates = lasagne.updates.rmsprop(
            critic_loss, critic_params, learning_rate=eta)

    # Instantiate a symbolic noise generator to use for training
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    srng = RandomStreams(seed=np.random.randint(2147462579, size=6))
    noise = srng.uniform((batchsize, 100))

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

    # We create an infinite supply of batches (as an iterable generator):
    print("Loading data...")
    batches = iterator(
        inputs, labels, batchsize, encoder, shuffle=True, length=patch_size,
        forever=True, alphabet_size=len(encoder.alphabet), padding=padding)
    # We iterate over epochs:
    generator_updates = 0
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
        samples = gen_fn(lasagne.utils.floatX(np.random.rand(42, 100)))

        plt.imsave('lsgan_text_{}.png'.format(epoch),
                   (samples.reshape(6, 7, len(encoder.alphabet), patch_size)
                           .transpose(0, 2, 1, 3)
                           .reshape(6*len(encoder.alphabet), 7*patch_size)),
                   origin='bottom',
                   cmap='jet')
        fig, axes = plt.subplots(1, 2)
        axes[0].plot(critic_losses)
        axes[1].plot(generator_losses)
        fig.savefig('lsgan_mnist_loss_epoch_{}'.format(epoch))
        plt.close('all')

        # After half the epochs, we start decaying the learn rate towards zero
        if epoch >= num_epochs // 2:
            progress = float(epoch) / num_epochs
            eta.set_value(lasagne.utils.floatX(initial_eta*2*(1 - progress)))
            plt.plot('')

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
