#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example employing Lasagne for piano roll using the MIDI files from classical
piano music
Wasserstein Generative Adversarial Networks
(WGANs, see https://arxiv.org/abs/1701.07875 for the paper and
https://github.com/martinarjovsky/WassersteinGAN for the "official" code).

Adapted from Jan Schlüter's code
https://gist.github.com/f0k/f3190ebba6c53887d598d03119ca2066
"""

from __future__ import print_function

import sys
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne

from data_processing import load_data
import matplotlib.pyplot as plt

import pdb


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
    from lasagne.nonlinearities import sigmoid
    # input: 100dim
    layer = InputLayer(shape=(None, 100), input_var=input_var)
    # fully-connected layer
    layer = batch_norm(DenseLayer(layer, 1024))
    # project and reshape
    layer = batch_norm(DenseLayer(layer, 128*32*32))
    layer = ReshapeLayer(layer, ([0], 128, 32, 32))
    # two fractional-stride convolutions
    layer = batch_norm(Deconv2DLayer(layer, 128, 5, stride=2, crop='same',
                                     output_size=64))
    layer = Deconv2DLayer(layer, 1, 5, stride=2, crop='same', output_size=128,
                          nonlinearity=sigmoid)
    print("Generator output:", layer.output_shape)
    return layer


def build_critic(input_var=None):
    from lasagne.layers import (InputLayer, Conv2DLayer, DenseLayer)
    try:
        from lasagne.layers.dnn import batch_norm_dnn as batch_norm
    except ImportError:
        from lasagne.layers import batch_norm
    from lasagne.nonlinearities import LeakyRectify
    lrelu = LeakyRectify(0.2)
    # input: (None, 1, 28, 28)
    layer = InputLayer(shape=(None, 1, 128, 128), input_var=input_var)
    # two convolutions
    layer = batch_norm(Conv2DLayer(layer, 64, 5, stride=2, pad='same',
                                   nonlinearity=lrelu))
    layer = batch_norm(Conv2DLayer(layer, 128, 5, stride=2, pad='same',
                                   nonlinearity=lrelu))
    # fully-connected layer
    layer = batch_norm(DenseLayer(layer, 1024, nonlinearity=lrelu))
    # output layer (linear and without bias)
    layer = DenseLayer(layer, 1, nonlinearity=None, b=None)
    print("Critic output:", layer.output_shape)
    return layer


def iterate_minibatches(inputs, batchsize, shuffle=True, forever=False,
                        length=128):
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

            if length > 0:
                data = []
                # select random slice from each piano roll
                for i in excerpt:
                    rand_start = np.random.randint(0, len(inputs[i]) - length)
                    data.append(inputs[i][rand_start:rand_start+length])
            else:
                data = inputs[excerpt]

            yield lasagne.utils.floatX(np.array(data))
        if not forever:
            break


def main(num_epochs=1000, epochsize=100, batchsize=64, initial_eta=1e-2,
         clip=0.01):
    # Load the dataset
    print("Loading data...")
    datapath = '/media/steampunkhd/rafaelvalle/datasets/MIDI/Piano'
    glob_file_str = '*.npy'
    n_pieces = 0  # 0 is equal to all pieces, unbalanced dataset
    crop = None  # crop = (32, 96)
    as_dict = False
    inputs, labels = load_data(datapath, glob_file_str, n_pieces, crop, as_dict,
                               patch_size=128)

    # Prepare Theano variables for inputs
    noise_var = T.matrix('noise')
    input_var = T.tensor4('inputs')

    # Create neural network model
    print("Building model and compiling functions...")
    generator = build_generator(noise_var)
    critic = build_critic(input_var)

    # Create expression for passing real data through the critic
    real_out = lasagne.layers.get_output(critic)
    # Create expression for passing fake data through the critic
    fake_out = lasagne.layers.get_output(
        critic, lasagne.layers.get_output(generator))

    # Create score expressions to be maximized (i.e., negative losses)
    generator_score = fake_out.mean()
    critic_score = real_out.mean() - fake_out.mean()

    # Create update expressions for training
    generator_params = lasagne.layers.get_all_params(generator, trainable=True)
    critic_params = lasagne.layers.get_all_params(critic, trainable=True)
    eta = theano.shared(lasagne.utils.floatX(initial_eta))
    generator_updates = lasagne.updates.rmsprop(
            -generator_score, generator_params, learning_rate=eta)
    critic_updates = lasagne.updates.rmsprop(
            -critic_score, critic_params, learning_rate=eta)

    # Clip critic parameters in a limited range around zero (except biases)
    for param in lasagne.layers.get_all_params(critic, trainable=True,
                                               regularizable=True):
        critic_updates[param] = T.clip(critic_updates[param], -clip, clip)

    # Instantiate a symbolic noise generator to use for training
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    srng = RandomStreams(seed=np.random.randint(2147462579, size=6))
    noise = srng.uniform((batchsize, 100))

    # Compile functions performing a training step on a mini-batch (according
    # to the updates dictionary) and returning the corresponding score:
    generator_train_fn = theano.function([], generator_score,
                                         givens={noise_var: noise},
                                         updates=generator_updates)
    critic_train_fn = theano.function([input_var], critic_score,
                                      givens={noise_var: noise},
                                      updates=critic_updates)

    # Compile another function generating some data
    gen_fn = theano.function([noise_var],
                             lasagne.layers.get_output(generator,
                                                       deterministic=True))

    # Finally, launch the training loop.
    print("Starting training...")
    # We create an infinite supply of batches (as an iterable generator):
    batches = iterate_minibatches(inputs, batchsize, shuffle=True,
                                  length=0, forever=True)
    # We iterate over epochs:
    generator_updates = 0
    for epoch in range(num_epochs):
        start_time = time.time()

        # In each epoch, we do `epochsize` generator updates. Usually, the
        # critic is updated 5 times before every generator update. For the
        # first 25 generator updates and every 500 generator updates, the
        # critic is updated 100 times instead, following the authors' code.
        critic_scores = []
        generator_scores = []
        for _ in range(epochsize):
            if (generator_updates < 25) or (generator_updates % 500 == 0):
                critic_runs = 100
            else:
                critic_runs = 5
            for _ in range(critic_runs):
                batch = next(batches)
                # reshape batch to proper dimensions
                batch = batch.reshape(
                    (batch.shape[0], 1, batch.shape[1], batch.shape[2]))
                critic_scores.append(critic_train_fn(batch))
            generator_scores.append(generator_train_fn())
            generator_updates += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  generator score:\t\t{}".format(np.mean(generator_scores)))
        print("  Wasserstein distance:\t\t{}".format(np.mean(critic_scores)))

        # And finally, we plot some generated data
        samples = gen_fn(lasagne.utils.floatX(np.random.rand(42, 100)))
        plt.imsave('images/wgan_proll/wgan_epoch{}.png'.format(epoch),
                   (samples.reshape(6, 7, 128, 128)
                           .transpose(0, 2, 1, 3)
                           .reshape(6*128, 7*128)).T,
                   cmap='gray')

        # After half the epochs, we start decaying the learn rate towards zero
        if epoch >= num_epochs // 2:
            progress = float(epoch) / num_epochs
            eta.set_value(lasagne.utils.floatX(initial_eta*2*(1 - progress)))

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('wgan_proll_gen.npz', *lasagne.layers.get_all_param_values(generator))
    # np.savez('wgan_proll_crit.npz', *lasagne.layers.get_all_param_values(critic))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a WGAN on Piano Rolls using Lasagne.")
        print("Usage: %s [EPOCHS [EPOCHSIZE]]" % sys.argv[0])
        print()
        print("EPOCHS: number of training epochs to perform (default: 1000)")
        print("EPOCHSIZE: number of generator updates per epoch (default: 100)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['num_epochs'] = int(sys.argv[1])
        if len(sys.argv) > 2:
            kwargs['epochsize'] = int(sys.argv[2])
        if len(sys.argv) > 3:
            kwargs['initial_eta'] = float(sys.argv[3])
        main(**kwargs)
