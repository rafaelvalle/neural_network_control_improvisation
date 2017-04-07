#!/usr/bin/python
from __future__ import print_function
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, argparse
import cPickle as pkl
import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import (get_all_layers, get_all_params, get_output_shape,
                            set_all_param_values, get_output)
from lasagne.utils import floatX

from models import build_generator

import pdb


def main(trial_folder, model_filename, g_arch, g_batch_norm=1, batch_size=8,
         n_steps=10):

    def interpolate(x, y, n_steps):
        step_size = 1.0/n_steps
        return [x + a*(y-x) for a in np.arange(0, 1+step_size, step_size)]

    def get_output_size(network, name):
        layer = [l for l in get_all_layers(network) if l.name == name]
        if len(layer):
            return get_output_shape(layer)[0][1]
        return 0

    def generate_output(gen_fn, noise, noise_size, cond=[], cond_size=0):
        if len(cond) and cond_size:
            return gen_fn(noise.reshape(1, noise_size),
                        cond.reshape(1, cond_size))[0, 0]
        else:
            return gen_fn(noise.reshape(1, noise_size))[0, 0]

    def plotting(ax, data):
            ax.imshow(data, aspect='auto', origin='bottom', cmap='gray')

    # load blank network and and use saved weights to update network parameters
    network = pkl.load(
        open(os.path.join(trial_folder, 'models/generator_blank.pkl'), "rb"))

    # instantiate noise and condition variables
    noise_var = T.fmatrix('noise')
    cond_var = T.fmatrix('condition')
    noise_size = get_output_size(network, 'g_in_noise')
    cond_size = get_output_size(network, 'g_in_condition')

    # create blank network given params
    network = build_generator(
        noise_var, noise_size, cond_var, cond_size, g_arch, g_batch_norm)
    with np.load(os.path.join(trial_folder, 'models/' + model_filename)) as f:
        parameters = [f['arr_%d' % i] for i in range(len(f.files))]
    set_all_param_values(network, parameters)

    # create batch of fixed noise vectors
    noise = floatX(np.random.rand(batch_size, noise_size))

    # create interpolations between fixed noise vectors
    noise_interp = [
        interpolate(noise[i], noise[i+1], n_steps)
        for i in range(batch_size-1)]

    # set proper generator function given conditional or unconditional
    if cond_size:
        # create batch of fixed condition vectors
        cond = floatX(np.eye(cond_size))
        cond_interp = [
            interpolate(cond[i], cond[i+1], n_steps)
            for i in range(len(cond)-1)]
        # create generator function
        gen_fn = theano.function(
            [noise_var, cond_var], get_output(network, deterministic=True))
    else:
        # create generator function
        gen_fn = theano.function(
            [noise_var], get_output(network, deterministic=True))

    print("Plotting fixed vectors")
    # plot noise vectors conditioned if applicable
    if cond_size:
        for i in range(batch_size):
            dim = int(np.sqrt(cond_size) + 1)
            fig, axes = plt.subplots(dim, dim, figsize=(16, 32))
            axes = axes.flatten()
            [plotting(axes[j], generate_output(gen_fn, noise[i], noise_size, cond[j], cond_size))
             for j in range(cond_size)]
            fig.tight_layout()
            fig.savefig('{}/samples/z_{}_conditioned.png'.format(
                trial_folder, i))
            plt.close('all')
    else:
        fig, axes = plt.subplots(1, batch_size, figsize=(16, 4))
        [plotting(axes[i], generate_output(gen_fn, noise[i], noise_size)) for i in range(batch_size)]
        fig.tight_layout()
        fig.savefig('{}/samples/z.png'.format(trial_folder))
        plt.close('all')


    print("Plotting interpolations")
    # plot interpolations, MUST REFACTOR
    if cond_size:
        """
        # fixed noise moving condition, find alternative visualization
        for i in range(batch_size):
            fig, axes = plt.subplots(cond_size-1, n_steps+1, figsize=(32, 32))
            [plotting(axes[j, k], generate_output(gen_fn, noise[i], noise_size, cond_interp[j][k], cond_size))
             for j in range(batch_size-1) for k in range(n_steps+1)]
            fig.tight_layout()
            fig.savefig(
                '{}/samples/z_{}_cond_interp.png'.format(trial_folder, i))
            plt.close('all')


        plt.imsave('{}/samples/z_{}_cond_interp.png'.format(trial_folder, i),
                   (samples.reshape(12, 12, alphabet_size, n_steps)
                           .transpose(0, 2, 1, 3)
                           .reshape(12*alphabet_size, 12*n_steps)),
                   origin='bottom',
                   cmap='gray')
        """
        # fixed condition moving noise
        for i in range(batch_size):
            fig, axes = plt.subplots(batch_size-1, n_steps+1, figsize=(32, 12))
            [plotting(axes[j, k], generate_output(gen_fn, noise_interp[j][k], noise_size, cond[i], cond_size))
             for j in range(batch_size-1) for k in range(n_steps+1)]
            fig.tight_layout()
            fig.savefig(
                '{}/samples/z_inter_cond_{}.png'.format(trial_folder, i))
            plt.close('all')
    else:
        # moving noise
        fig, axes = plt.subplots(batch_size-1, n_steps+1, figsize=(32, 12))
        [plotting(axes[j, k], generate_output(gen_fn, noise_interp[j][k], noise_size))
         for j in range(batch_size-1) for k in range(n_steps+1)]
        fig.tight_layout()
        fig.savefig(
            '{}/samples/z_inter.png'.format(trial_folder))
        plt.close('all')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=("Generates output by traversing the latent space Z and" +
                     "condition vector using a saved Generator"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("trial_folder", type=str,
                        help="Path of trial folder")
    parser.add_argument("model_filename", type=str,
                        help="Filename of model in trial_folder")
    parser.add_argument("g_arch", type=str,
                        help="Name of generator architechture")
    parser.add_argument("-b", type=int, default=1,
                        help="Generator has batch norm or not")

    args = parser.parse_args()
    print(args)
    main(args.trial_folder, args.model_filename, args.g_arch)
