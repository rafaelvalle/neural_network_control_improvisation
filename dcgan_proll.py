#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne

from data_processing import load_data

import pdb
import matplotlib.pyplot as plt

noise_size = 100
class Deconv2DLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_filters, filter_size, stride=1, pad=0,
                 nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
        super(Deconv2DLayer, self).__init__(incoming, **kwargs)
        self.num_filters = num_filters
        self.filter_size = lasagne.utils.as_tuple(filter_size, 2, int)
        self.stride = lasagne.utils.as_tuple(stride, 2, int)
        self.pad = lasagne.utils.as_tuple(pad, 2, int)
        self.W = self.add_param(
            lasagne.init.Orthogonal(),
            (self.input_shape[1], num_filters) + self.filter_size, name='W')
        self.b = self.add_param(lasagne.init.Constant(0),
                                (num_filters,), name='b')
        if nonlinearity is None:
            nonlinearity = lasagne.nonlinearities.identity
        self.nonlinearity = nonlinearity

    def get_output_shape_for(self, input_shape):
        shape = tuple(i*s - 2*p + f - 1
                      for i, s, p, f in zip(input_shape[2:],
                                            self.stride,
                                            self.pad,
                                            self.filter_size))
        return (input_shape[0], self.num_filters) + shape

    def get_output_for(self, input, **kwargs):
        op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(
            imshp=self.output_shape,
            kshp=(self.input_shape[1], self.num_filters) + self.filter_size,
            subsample=self.stride, border_mode=self.pad)
        conved = op(self.W, input, self.output_shape[2:])
        if self.b is not None:
            conved += self.b.dimshuffle('x', 0, 'x', 'x')
        return self.nonlinearity(conved)


def build_generator(input_var=None):
    from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer, batch_norm
    from lasagne.nonlinearities import sigmoid
    # noise input
    layer = InputLayer(shape=(None, noise_size), input_var=input_var)
    # fully-connected layer
    layer = batch_norm(DenseLayer(layer, 1024))
    # project and reshape
    layer = batch_norm(DenseLayer(layer, 128*16*16))
    layer = ReshapeLayer(layer, ([0], 128, 16, 16))
    # two fractional-stride convolutions
    layer = batch_norm(Deconv2DLayer(layer, 64, 5, stride=2, pad=2))
    layer = Deconv2DLayer(layer, 1, 5, stride=2, pad=2, nonlinearity=sigmoid)
    print("Generator output:", layer.output_shape)
    return layer


def build_discriminator(input_var=None):
    from lasagne.layers import (InputLayer, DenseLayer, batch_norm)
    from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer  # override
    from lasagne.nonlinearities import LeakyRectify, sigmoid
    lrelu = LeakyRectify(0.2)
    # input: (None, 1, 64, 64)
    layer = InputLayer(shape=(None, 1, 64, 64), input_var=input_var)
    # two convolutions
    layer = batch_norm(
        Conv2DLayer(layer, 64, 5, stride=2, pad=2, nonlinearity=lrelu))
    layer = batch_norm(
        Conv2DLayer(layer, 128, 5, stride=2, pad=2, nonlinearity=lrelu))
    # fully-connected layer
    layer = batch_norm(DenseLayer(layer, 1024, nonlinearity=lrelu))
    # output layer
    layer = DenseLayer(layer, 1, nonlinearity=sigmoid)
    print("Discriminator output:", layer.output_shape)
    return layer


def iterate_minibatches(inputs, batchsize, length=64, shuffle=True,
                        forever=False):
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
                rand_start = np.random.randint(0, len(inputs[i]) - length)
                data.append(inputs[i][rand_start:rand_start+length])

            yield np.array(data)
        if not forever:
            break


def main(num_epochs=200, initial_eta=1e-4):
    # Load the dataset
    print("Loading data...")
    datapath = '/media/steampunkhd/rafaelvalle/datasets/MIDI/Piano'
    glob_file_str = '*.npy'
    n_pieces = 0  # 0 is equal to all pieces, unbalanced dataset
    crop = (32, 96)
    as_dict = False
    dataset = load_data(datapath, glob_file_str, n_pieces, crop, as_dict)

    # scale to [0, 1]
    # dataset = (dataset + 1) * 0.5

    # Prepare Theano variables for inputs and targets
    noise_var = T.matrix('noise')
    input_var = T.tensor4('inputs')

    # Create neural network model
    print("Building model and compiling functions...")
    generator = build_generator(noise_var)
    discriminator = build_discriminator(input_var)

    # Create expression for passing real data through the discriminator
    real_out = lasagne.layers.get_output(discriminator)
    # Create expression for passing fake data through the discriminator
    fake_out = lasagne.layers.get_output(
        discriminator, lasagne.layers.get_output(generator))

    # Create loss expressions
    generator_loss = lasagne.objectives.binary_crossentropy(fake_out, 1).mean()
    discriminator_loss = (
        lasagne.objectives.binary_crossentropy(real_out, 1) +
        lasagne.objectives.binary_crossentropy(fake_out, 0)).mean()

    # Create update expressions for training
    generator_params = lasagne.layers.get_all_params(generator, trainable=True)
    discriminator_params = lasagne.layers.get_all_params(discriminator, trainable=True)
    eta = theano.shared(lasagne.utils.floatX(initial_eta))
    updates = lasagne.updates.adam(
        generator_loss, generator_params, learning_rate=eta, beta1=0.5)
    updates.update(lasagne.updates.adam(
            discriminator_loss, discriminator_params, learning_rate=eta, beta1=0.5))

    # Instantiate a symbolic noise generator to use for training
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    srng = RandomStreams(seed=np.random.randint(2147462579, size=6))
    noise = srng.uniform((batchsize, 100))

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var],
                               [(real_out > .5).mean(), (fake_out < .5).mean()],
                               givens={noise_var: noise,
                               updates=updates)


    # Compile another function generating some data
    gen_fn = theano.function([noise_var],
                             lasagne.layers.get_output(generator,
                                                       deterministic=True))

    print("Starting training...")
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(dataset, 64):
            batch = lasagne.utils.floatX(batch)
            # reshape batch to proper dimensions
            batch = batch.reshape(
                (batch.shape[0], 1, batch.shape[1], batch.shape[2]))
            train_err += np.array(train_fn(batch))
            train_batches += 1


        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{}".format(train_err / train_batches))

        # And finally, we plot some generated data
        samples = gen_fn(lasagne.utils.floatX(np.random.rand(42, noise_size)))
        plt.imsave('images/dcgan_proll/mnist_samples_epoch{}.png'.format(epoch),
                    (samples.reshape(6, 7, 64, 64)
                            .transpose(0, 2, 1, 3)
                            .reshape(6*64, 7*64)).T,
                    cmap='gray')

        # After half the epochs, start decaying the learning rate towards zero
        if epoch >= num_epochs // 2:
            progress = float(epoch) / num_epochs
            eta.set_value(lasagne.utils.floatX(initial_eta*2*(1 - progress)))

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('proll_gen.npz', *lasagne.layers.get_all_param_values(generator))
    # np.savez('proll_disc.npz', *lasagne.layers.get_all_param_values(discriminator))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a DCGAN on MNIST using Lasagne.")
        print("Usage: %s [EPOCHS]" % sys.argv[0])
        print()
        print("EPOCHS: number of training epochs to perform (default: 100)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['num_epochs'] = int(sys.argv[1])
        main(**kwargs)
