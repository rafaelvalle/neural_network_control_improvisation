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

from models import build_generator, build_generator_lstm, build_critic
from data_processing import load_proll_data, load_text_data, encode_labels
from text_utils import textEncoder
import pdb

# add (real, fake) pair as input


def iterate_minibatches_proll(inputs, labels, batch_size, shuffle=True,
                              forever=True, length=128):
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


def iterate_minibatches_text(inputs, labels, batch_size, encoder, shuffle=True,
                             forever=True, length=128, alphabet_size=128,
                             padding=None):
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
                else:
                    data.append(cur_data)
            # scale to [-1, 1]
            data = np.array(data)
            data += data.min()
            data /= data.max()
            data = (data * 2) - 1
            yield lasagne.utils.floatX(data), labels[excerpt]

        if not forever:
            break


def build_functions(critic, generator, clip, batch_size, input_var, noise_var,
                    cond_var, c_eta, g_eta, noise_size, loss_type,
                    build_grads=False, n_steps=None, g_arch=None):

    # instantiate a symbolic noise generator to use in training
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    srng = RandomStreams(seed=np.random.randint(2147462579, size=6))
    if g_arch.startswith("lstm"):
        noise = srng.normal((batch_size, n_steps, noise_size))
    else:
        noise = srng.normal((batch_size, noise_size))
    if loss_type == 'iwgan':
        # alpha = np.random.uniform(0, 1, (batch_size, 1, 1))
        alpha = srng.uniform((batch_size, 1, 1, 1), low=0., high=1.)

    # prepare symbolic input variables given condition
    if cond_var:
        gen_input = [cond_var]
        cri_input = [input_var, cond_var]
        samp_input = [noise_var, cond_var]
    else:
        gen_input = []
        cri_input = [input_var]
        samp_input = [noise_var]

    # get params of each network
    generator_params = lasagne.layers.get_all_params(generator, trainable=True)
    critic_params = lasagne.layers.get_all_params(critic, trainable=True)

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
    if loss_type in ('wgan', 'iwgan'):
        generator_score = -fake_out.mean()
        critic_score = fake_out.mean() - real_out.mean()
        if loss_type == 'iwgan':
            """
            differences = fake_inputs - real_inputs
            interpolates = real_inputs + (alpha*differences)
            gradients = tf.gradients(Discriminator(interpolates)[0], [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2]))
            gradient_penalty = tf.reduce_mean((slopes-1.)**2)
            disc_cost += LAMBDA*gradient_penalty
            """

            LAMBDA = 10
            differences = lasagne.layers.get_output(generator) - input_var
            interpolates = input_var + (alpha*differences)
            gradients = theano.grad(real_out[0], interpolates)[0]
            # slopes = T.sqrt(T.sum(T.sqr(gradients), axis=(1, 2)))
            # gradient_penalty = T.mean((slopes-1.)**2)
            # critic_score += LAMBDA * gradient_penalty
    elif loss_type == 'lsgan':
        # a, b, c = -1, 1, 0  # Equation (8) in the paper
        a, b, c = 0, 1, 1  # Equation (9) in the paper
        generator_score = lasagne.objectives.squared_error(fake_out, c).mean()
        critic_score = (lasagne.objectives.squared_error(real_out, b).mean() +
                        lasagne.objectives.squared_error(fake_out, a).mean())

    # Create update expressions for training
    generator_updates = lasagne.updates.rmsprop(
            generator_score, generator_params, learning_rate=g_eta)
    critic_updates = lasagne.updates.rmsprop(
            critic_score, critic_params, learning_rate=c_eta)

    if loss_type != 'iwgan' and clip != 0:
        # Clip critic parameters in a limited range around value (except biases)
        for param in lasagne.layers.get_all_params(critic, trainable=True,
                                                   regularizable=True):
            critic_updates[param] = T.clip(critic_updates[param], -clip, clip)


    # train functions
    generator_train_fn = theano.function(gen_input, [generator_score, gradients],
                                         givens={noise_var: noise},
                                         updates=generator_updates)
    critic_train_fn = theano.function(cri_input, critic_score,
                                      givens={noise_var: noise},
                                      updates=critic_updates)

    # Compile another function generating some data
    gen_fn = theano.function(samp_input,
                             lasagne.layers.get_output(generator,
                                                       deterministic=True))
    # compile functions for looking at gradients
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


def main(data_type, c_arch, g_arch, num_epochs, epoch_size, batch_size,
         c_initial_eta, g_initial_eta, clip, noise_size, boolean, conditional,
         c_batch_norm, g_batch_norm, c_iters, cl_iters, loss_type, name,
         cl_freq, weight_decay, save_model_every):
    # Load the data according to datatype
    print("Loading data...")
    if data_type == 'text':
        datapaths = (
            '/media/steampunkhd/rafaelvalle/datasets/TEXT/ag_news_csv/train.csv',)
            #'/media/steampunkhd/rafaelvalle/datasets/TEXT/ag_news_csv/test.csv')

        data_cols = (2, 1)
        label_cols = (0, 0)
        n_pieces = 0  # 0 is equal to all pieces, unbalanced dataset
        as_dict = False
        n_steps = 128
        padding = 'repeat'
        inputs, labels = load_text_data(
            datapaths, data_cols, label_cols, n_pieces, as_dict,
            patch_size=n_steps)
        iterator = iterate_minibatches_text
        # alphabet = list("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}")
        alphabet = [chr(x) for x in range(127)]
        encoder = textEncoder(alphabet)
        alphabet_size = len(encoder.alphabet) + 1
        pkl.dump(encoder, open("encdec.pkl", "wb"))
        i_len = 128
    elif data_type == 'proll':
        datapath = '/media/steampunkhd/rafaelvalle/datasets/MIDI/JazzSolos'
        glob_file_str = '*.npy'
        n_pieces = 0  # 0 is equal to all pieces, unbalanced dataset
        crop = None  # (32, 96)
        alphabet_size = 128
        as_dict = False
        n_steps = 128
        i_len = 0
        inputs, labels = load_proll_data(
            datapath, glob_file_str, n_pieces, crop, as_dict,
            patch_size=n_steps)
        if boolean:
            inputs += inputs.min()
            inputs /= inputs.max()
            inputs = inputs.astype(bool)
            # project to [-1, 1]
            inputs = lasagne.utils.floatX((inputs * 2) - 1)
        iterator = iterate_minibatches_proll
    else:
        raise Exception("Datatype {} not supported".format(data_type))

    # encode labels
    labels = encode_labels(labels, one_hot=True).astype(np.float32)
    print("Dataset shape {}".format(inputs.shape))

    # create fixed conditions and noise for output evaluation
    n_samples = 12 * 12
    if conditional:
        fixed_condition = lasagne.utils.floatX(np.eye(labels.shape[1])[
            np.repeat(np.arange(labels.shape[1]), 1+(n_samples / labels.shape[1]))])
        fixed_condition = fixed_condition[:n_samples]

    # Prepare theano variables
    if g_arch.startswith('lstm'):
        fixed_noise = lasagne.utils.floatX(
            np.random.rand(n_samples, n_steps, noise_size))
        noise_var = T.ftensor3('noise')
        build_gen = build_generator_lstm
    else:
        build_gen = build_generator
        fixed_noise = lasagne.utils.floatX(np.random.rand(n_samples, noise_size))
        noise_var = T.fmatrix('noise')

    input_var = T.ftensor4('inputs')
    cond_var = None
    if conditional:
        cond_var = T.fmatrix('condition')

    # learning rates
    c_eta = theano.shared(lasagne.utils.floatX(c_initial_eta))
    g_eta = theano.shared(lasagne.utils.floatX(g_initial_eta))

    # Create neural network model
    print("Building model and compiling functions...")
    generator = build_gen(
        noise_var, noise_size, cond_var, labels.shape[1], g_arch, g_batch_norm,
        batch_size=None, n_steps=n_steps)
    critic = build_critic(
        input_var, cond_var, labels.shape[1], c_arch, c_batch_norm,
        loss_type=loss_type)
    print("Generator {} #params {}\nModel {}".format(
        g_arch, lasagne.layers.count_params(generator),
        lasagne.layers.get_all_layers(generator)))
    print("Critic {} #params {}\nModel {}".format(
        c_arch, lasagne.layers.count_params(critic),
        lasagne.layers.get_all_layers(critic)))

    # Build train and sampling functions
    g_train_fn, c_train_fn, g_gen_fn, c_grad_fn, g_grad_fn = build_functions(
        critic, generator, clip, batch_size, input_var, noise_var, cond_var,
        c_eta, g_eta, noise_size, loss_type, n_steps=n_steps, g_arch=g_arch)

    # Create an infinite supply of batches (as an iterable generator)
    if data_type == 'text':
        batches = iterator(
            inputs, labels, batch_size, encoder, shuffle=True, length=i_len,
            forever=True, alphabet_size=alphabet_size, padding=padding)
    elif data_type == 'proll':
        batches = iterator(
            inputs, labels, batch_size, shuffle=True, length=i_len,
            forever=True)

    # set variables for storing scores
    if not epoch_size:
        epoch_size = len(inputs) / batch_size
    generator_iterations = 0
    epoch_critic_scores = []
    epoch_generator_scores = []
    print("Starting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        critic_scores = []
        generator_scores = []
        for _ in tqdm(range(epoch_size)):
            if (generator_iterations < 25) or (generator_iterations % cl_freq== 0):
                critic_runs = cl_iters # 100
            else:
                critic_runs = c_iters # 5
            for _ in range(critic_runs):
                batch_in, batch_cond = next(batches)
                # reshape batch to proper dimensions
                batch_in = batch_in.reshape(
                    (batch_in.shape[0], 1, batch_in.shape[1], batch_in.shape[2]))
                if cond_var:
                    critic_scores.append(c_train_fn(batch_in, batch_cond))
                else:
                    loss, grads = c_train_fn(batch_in)
                    # critic_scores.append(c_train_fn(batch_in))
            if cond_var:
                generator_scores.append(g_train_fn(batch_cond))
            else:
                generator_scores.append(g_train_fn())
            generator_iterations += 1

        # Then we print the results for this epoch:
        print("""Epoch {} of {} took {:.3f}s\t
              Critic Loss {} \t Generator Loss {}""".format(
              epoch + 1, num_epochs, time.time() - start_time,
              np.mean(critic_scores), np.mean(generator_scores)))

        epoch_critic_scores.append(np.mean(critic_scores))
        epoch_generator_scores.append(np.mean(generator_scores))

        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        axes[0].set_title('Loss(C)')
        axes[0].plot(epoch_critic_scores)
        axes[1].set_title('Loss(G)')
        axes[1].plot(epoch_generator_scores)
        fig.tight_layout()
        fig.savefig('images/{}_{}/{}_g_updates.png'.format(
            loss_type, data_type, name))
        plt.close('all')

        # plot and create midi from generated data
        if cond_var:
            samples = g_gen_fn(fixed_noise, fixed_condition)
        else:
            samples = g_gen_fn(fixed_noise)
        plt.imsave('images/{}_{}/{}_gits{}.png'.format(loss_type, data_type, name, epoch),
                   (samples.reshape(12, 12, alphabet_size, n_steps)
                           .transpose(0, 2, 1, 3)
                           .reshape(12*alphabet_size, 12*n_steps)),
                   origin='bottom',
                   cmap='gray')
        np.save('{1}/{0}_{1}/{2}_gits{3}.npy'.format(loss_type, data_type, name, epoch), samples)

        # After half the epochs, we start decaying the learn rate towards zero
        if weight_decay:
            if epoch >= num_epochs // 2:
                progress = float(epoch) / num_epochs
                c_eta.set_value(lasagne.utils.floatX(
                    c_initial_eta*2*(1 - progress)))
                g_eta.set_value(lasagne.utils.floatX(
                    g_initial_eta*2*(1 - progress)))

        if (epoch % save_model_every) == 0:
            np.savez('models/{}_{}_gen_{}_{}.npz'.format(loss_type, data_type, name, epoch),
                     *lasagne.layers.get_all_param_values(generator))
            np.savez('models/{}_{}_crit_{}_{}.npz'.format(loss_type, data_type, name, epoch),
                     *lasagne.layers.get_all_param_values(critic))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Trains GAN on Sequential data and generates output",
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
    parser.add_argument("--name", type=str, default='name',
                        help="Name to include in saved files")
    parser.add_argument("--cl_freq", type=int, default=500,
                        help="Frequency of large updates")
    parser.add_argument("--decay", type=int, default=1,
                        help="Apply weight decay?")
    parser.add_argument("-s", "--save_model_every", type=int, default=9,
                        help="Save model every?")

    args = parser.parse_args()
    print(args)
    main(args.datatype, args.critic, args.generator, args.n_epochs,
         args.epoch_size, args.bs, args.clr, args.glr, args.clip,
         args.noise_size, args.boolean, args.condition, args.cbn, args.gbn,
         args.c_iters, args.cl_iters, args.loss_type, args.name,
         args.cl_freq, args.decay, args.save_model_every)
