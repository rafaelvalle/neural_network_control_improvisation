#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example employing Lasagne for sequence generation using CNN
Crepe
https://github.com/Azure/Cortana-Intelligence-Gallery-Content/blob/master/Tutorials/Deep-Learning-for-Text-Classification-in-Azure/python/03%20-%20Crepe%20-%20Amazon%20(advc).py

Wasserstein Generative Adversarial Networks
(WGANs, see https://arxiv.org/abs/1701.07875 for the paper and
https://github.com/martinarjovsky/WassersteinGAN for the "official" code).
"""

from __future__ import print_function
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm
import time
import argparse
import cPickle as pkl

import numpy as np
np.random.seed(1234)
import theano
import theano.tensor as T
import lasagne

from models import build_generator, build_critic
from data_processing import load_proll_data, load_text_data, encode_labels
from text_utils import textEncoder
import pdb

# add (real, fake) pair as input


def iterate_minibatches_proll(inputs, labels, batchsize, shuffle=True,
                              forever=True, length=128):
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
                    rand_start = np.random.randint(
                        0, inputs[i].shape[1] - length)
                    data.append(inputs[i][:, rand_start:rand_start+length])
            else:
                data = inputs[excerpt]

            yield lasagne.utils.floatX(np.array(data)), labels[excerpt]

        if not forever:
            break


def iterate_minibatches_text(inputs, labels, batchsize, encoder, shuffle=True,
                             forever=True, length=128, alphabet_size=128,
                             padding=None):
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
                    if padding is None:
                        # ignore examples shorter than length
                        continue
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
                    cur_data = z
                    data.append(cur_data)
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


def build_functions(critic, generator, clip, batchsize, input_var, noise_var,
                    cond_var, c_eta, g_eta, noise_size, loss_type,
                    build_grads=False):
    # Create expression for passing real data through the critic
    real_out = lasagne.layers.get_output(critic)
    # Create expression for passing fake data through the critic
    if cond_var:
        print("Conditional Network")
        d_in_layer = [l for l in lasagne.layers.get_all_layers(critic)
                      if l.name == 'd_in_data'][0]
        d_cond_layer = [l for l in lasagne.layers.get_all_layers(critic)
                        if l.name == 'd_in_condition'][0]
        fake_out = lasagne.layers.get_output(
            critic,
            inputs={d_in_layer: lasagne.layers.get_output(generator),
                    d_cond_layer: cond_var})
    else:
        print("Unconditional Network")
        fake_out = lasagne.layers.get_output(
            critic, lasagne.layers.get_output(generator))

    # Create score expressions to be maximized (i.e., negative losses)
    if loss_type == 'wgan':
        generator_score = fake_out.mean()
        critic_score = real_out.mean() - fake_out.mean()
    elif loss_type == 'lsgan':
        # a, b, c = -1, 1, 0  # Equation (8) in the paper
        a, b, c = 0, 1, 1  # Equation (9) in the paper
        generator_score = lasagne.objectives.squared_error(fake_out, c).mean()
        critic_score = (lasagne.objectives.squared_error(real_out, b).mean() +
                        lasagne.objectives.squared_error(fake_out, a).mean())

    # Create update expressions for training
    generator_params = lasagne.layers.get_all_params(generator, trainable=True)
    critic_params = lasagne.layers.get_all_params(critic, trainable=True)
    generator_updates = lasagne.updates.rmsprop(
            -generator_score, generator_params, learning_rate=g_eta)
    critic_updates = lasagne.updates.rmsprop(
            -critic_score, critic_params, learning_rate=c_eta)

    if clip != 0:
        # Clip critic parameters in a limited range around value (except biases)
        for param in lasagne.layers.get_all_params(critic, trainable=True,
                                                regularizable=True):
            critic_updates[param] = T.clip(critic_updates[param], -clip, clip)

    # Instantiate a symbolic noise generator to use for training
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    srng = RandomStreams(seed=np.random.randint(2147462579, size=6))
    noise = srng.normal((batchsize, noise_size))

    # Compile functions performing a training step on a mini-batch (according
    # to the updates dictionary) and returning the corresponding score:
    if cond_var:
        gen_input = [cond_var]
        cri_input = [input_var, cond_var]
        samp_input = [noise_var, cond_var]
    else:
        gen_input = []
        cri_input = [input_var]
        samp_input = [noise_var]
    generator_train_fn = theano.function(gen_input,
                                         generator_score,
                                         givens={noise_var: noise},
                                         updates=generator_updates)
    critic_train_fn = theano.function(cri_input,
                                      critic_score,
                                      givens={noise_var: noise},
                                      updates=critic_updates)

    # Compile another function generating some data
    gen_fn = theano.function(samp_input,
                             lasagne.layers.get_output(generator,
                                                       deterministic=True))
    if build_grads:
        critic_grad_tn = theano.grad(critic_score, critic_params)
        critic_grad_fn = theano.function(cri_input,
                                         critic_grad_tn,
                                         givens={noise_var: noise})
        generator_grad_tn = theano.grad(generator_score, generator_params)
        generator_grad_fn = theano.function(gen_input,
                                            generator_grad_tn,
                                            givens={noise_var: noise})
    else:
        critic_grad_fn = None
        generator_grad_fn = None

    return (generator_train_fn, critic_train_fn, gen_fn, critic_grad_fn,
            generator_grad_fn)


def main(datatype, c_arch, g_arch, num_epochs, epoch_size, batchsize,
         c_initial_eta, g_initial_eta, clip, noise_size, boolean, conditional,
         c_batch_norm, g_batch_norm, c_iters, cl_iters, loss_type):
    # Load the data according to datatype
    print("Loading data...")
    if datatype == 'text':
        datapaths = (
            '/media/steampunkhd/rafaelvalle/datasets/TEXT/ag_news_csv/train.csv',)
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

    elif datatype == 'proll':
        datapath = '/media/steampunkhd/rafaelvalle/datasets/MIDI/JazzSolos'
        glob_file_str = '*.npy'
        n_pieces = 0  # 0 is equal to all pieces, unbalanced dataset
        crop = None  # (32, 96)
        as_dict = False
        inputs, labels = load_proll_data(
            datapath, glob_file_str, n_pieces, crop, as_dict, patch_size=128)
        if boolean:
            inputs += inputs.min()
            inputs /= inputs.max()
            inputs = inputs.astype(bool)
            # project to [-1, 1]
            inputs = lasagne.utils.floatX((inputs * 2) - 1)
            iterator = iterate_minibatches_proll
    else:
        raise Exception("Datatype {} not supported".format(datatype))

    # encode labels
    labels = encode_labels(labels, one_hot=True).astype(np.float32)
    print("Dataset shape {}".format(inputs.shape))

    # create fixed conditions and noise for output evaluation
    n_samples = 12 * 12
    if conditional:
        fixed_condition = lasagne.utils.floatX(np.eye(labels.shape[1])[
            np.repeat(np.arange(labels.shape[1]), 1+(n_samples / labels.shape[1]))])
        fixed_condition = fixed_condition[:n_samples]
    fixed_noise = lasagne.utils.floatX(np.random.rand(n_samples, noise_size))

    # Prepare Theano variables
    input_var = T.ftensor4('inputs')
    noise_var = T.fmatrix('noise')
    cond_var = None
    if conditional:
        cond_var = T.fmatrix('condition')

    # learning rates
    c_eta = theano.shared(lasagne.utils.floatX(c_initial_eta))
    g_eta = theano.shared(lasagne.utils.floatX(g_initial_eta))

    # Create neural network model
    print("Building model and compiling functions...")
    generator = build_generator(
        noise_var, noise_size, cond_var, labels.shape[1], g_arch, g_batch_norm)
    critic = build_critic(
        input_var, cond_var, labels.shape[1], c_arch, c_batch_norm)
    print("Generator {} #params {}\nModel {}".format(
        g_arch, lasagne.layers.count_params(generator),
        lasagne.layers.get_all_layers(generator)))
    print("Critic {} #params {}\nModel {}".format(
        c_arch, lasagne.layers.count_params(critic),
        lasagne.layers.get_all_layers(critic)))

    # Build train and sampling functions
    g_train_fn, c_train_fn, g_gen_fn, c_grad_fn, g_grad_fn = build_functions(
        critic, generator, clip, batchsize, input_var, noise_var, cond_var,
        c_eta, g_eta, noise_size, loss_type)

    # Create an infinite supply of batches (as an iterable generator)
    if datatype == 'text':
        batches = iterator(
            inputs, labels, batchsize, encoder, shuffle=True, length=patch_size,
            forever=True, alphabet_size=len(encoder.alphabet), padding=padding)
    elif datatype == 'proll':
        batches = iterator(
            inputs, labels, batchsize, shuffle=True, length=128, forever=True)

    # set variables for storing scores
    if not epoch_size:
        epoch_size = len(inputs) / batchsize
    generator_updates = 0
    epoch_critic_scores = []
    epoch_generator_scores = []
    print("Starting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        critic_scores = []
        generator_scores = []
        for _ in tqdm(range(epoch_size)):
            if (generator_updates < 25) or (generator_updates % 500 == 0):
                critic_runs = cl_iters # 100
            else:
                critic_runs = c_iters # 5
            for _ in range(critic_runs):
                batch_in, batch_cond = next(batches)
                # reshape batch to proper dimensions
                batch_in = batch_in.reshape(
                    (batch_in.shape[0], 1, batch_in.shape[1], batch_in.shape[2]))
                if cond_var:
                    train_input = [batch_in, batch_cond]
                else:
                    train_input = batch_in
                critic_scores.append(c_train_fn(train_input))
            if cond_var:
                generator_scores.append(g_train_fn(batch_cond))
            else:
                generator_scores.append(g_train_fn())
            generator_updates += 1

        # Then we print the results for this epoch:
        print("""Epoch {} of {} took {:.3f}s\t
              Critic Loss {} \t Generator Loss {}""".format(
              epoch + 1, num_epochs, time.time() - start_time,
              np.mean(critic_scores), np.mean(generator_scores)))

        epoch_critic_scores.append(np.mean(generator_scores))
        epoch_generator_scores.append(np.mean(critic_scores))

        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        axes[0].set_title('Loss(C)')
        axes[0].plot(epoch_critic_scores)
        axes[1].set_title('Loss(G)')
        axes[1].plot(epoch_generator_scores)
        fig.tight_layout()
        fig.savefig('images/wcgan_{}/g_updates{}'.format(datatype, epoch))
        plt.close('all')

        # plot and create midi from generated data
        if cond_var:
            samples = g_gen_fn(fixed_noise, fixed_condition)
        else:
            samples = g_gen_fn(fixed_noise)
        plt.imsave('images/wcgan_{}/wcgan_gits{}.png'.format(datatype, epoch),
                   (samples.reshape(12, 12, len(encoder.alphabet), patch_size)
                           .transpose(0, 2, 1, 3)
                           .reshape(12*len(encoder.alphabet), 12*patch_size)),
                   origin='bottom',
                   cmap='jet')
        if datatype == 'text':
            np.save('text/wcgan_text/wcgan_gits{}.npy'.format(epoch), samples)
        else:
            np.save('midi/wcgan_proll/wcgan_gits{}.npy'.format(epoch), samples)

        # After half the epochs, we start decaying the learn rate towards zero
        if epoch >= num_epochs // 2:
            progress = float(epoch) / num_epochs
            c_eta.set_value(lasagne.utils.floatX(
                c_initial_eta*2*(1 - progress)))
            g_eta.set_value(lasagne.utils.floatX(
                g_initial_eta*2*(1 - progress)))

        if (epoch % 9) == 0:
            np.savez('wcgan_{}_gen_{}.npz'.format(datatype, epoch),
                     *lasagne.layers.get_all_param_values(generator))
            np.savez('wcgan_{}_crit_{}.npz'.format(datatype, epoch),
                     *lasagne.layers.get_all_param_values(critic))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Trains WCGAN on Sequential data and generates output",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("datatype", type=str, default='text',
                        help="Datatype <text> or <proll>")
    parser.add_argument("critic", type=str, default='crepe',
                        help="Critic architecture from models")
    parser.add_argument("generator", type=str, default='mnist',
                        help="Generator architecture from models")
    parser.add_argument("-n", "--n_epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("-e", "--epoch_size", type=int, default=0,
                        help="Epoch Size")
    parser.add_argument("-m", "--bs", type=int, default=128,
                        help="Mini-Batch Size")
    parser.add_argument("--clr", type=float, default=1e-4,
                        help="Critic Learning Rate")
    parser.add_argument("--glr", type=float, default=1e-4,
                        help="Generator Learning Rate")
    parser.add_argument("-p", "--clip", type=float, default=0.01,
                        help="Clip weights")
    parser.add_argument("-z", "--noise_size", type=int, default=256,
                        help="Size of noise vector")
    parser.add_argument("-b", "--boolean", type=int, default=0,
                        help="Data as boolean")
    parser.add_argument("-c", "--condition", type=int, default=0,
                        help="Include labels/conditioning")
    parser.add_argument("--cbn", type=int, default=0,
                        help="Batch norm critic")
    parser.add_argument("--gbn", type=int, default=0,
                        help="Batch Norm generator")
    parser.add_argument("--c_iters", type=int, default=5,
                        help="Discriminator iters")
    parser.add_argument("--cl_iters", type=int, default=100,
                        help="Discriminator large iters")
    parser.add_argument("-l", "--loss_type", type=str, default='wgan',
                        help="Loss type <wgan, lsgan>")

    args = parser.parse_args()
    print(args)
    main(args.datatype, args.critic, args.generator, args.n_epochs,
         args.epoch_size, args.bs, args.clr, args.glr, args.clip,
         args.noise_size, args.boolean, args.condition, args.cbn, args.gbn,
         args.c_iters, args.cl_iters, args.loss_type)
