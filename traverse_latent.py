#!/usr/bin/python
from __future__ import print_function
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import InputLayer, DenseLayer, ReshapeLayer
from lasagne.layers import get_all_params, set_all_param_values, get_output
from lasagne.layers import TransposedConv2DLayer as Deconv2DLayer
try:
    from lasagne.layers.dnn import batch_norm_dnn as batch_norm
except ImportError:
    from lasagne.layers import batch_norm
from lasagne.nonlinearities import tanh

import pdb

batch_size = 32
noise_size = 1024
noise_var = T.fmatrix('noise')

print("MNIST generator")
layer = InputLayer(shape=(None, noise_size), input_var=noise_var)
layer = batch_norm(DenseLayer(layer, 1024))
layer = batch_norm(DenseLayer(layer, 1024*8*8))
layer = ReshapeLayer(layer, ([0], 1024, 8, 8))
layer = batch_norm(Deconv2DLayer(
    layer, 256, 5, stride=2, crop='same', output_size=16))
layer = batch_norm(Deconv2DLayer(
    layer, 256, 5, stride=2, crop='same', output_size=32))
layer = batch_norm(Deconv2DLayer(
    layer, 256, 5, stride=2, crop='same', output_size=64))
layer = Deconv2DLayer(
    layer, 1, 5, stride=2, crop='same', output_size=128,
    nonlinearity=tanh)
params = get_all_params(layer, trainable=False)

# use saved weights
with np.load('lsgan_proll_gen.npz') as f:
    parameters = [f['arr_%d' % i] for i in range(len(f.files))]
set_all_param_values(layer, parameters)

gen_fn = theano.function([noise_var], get_output(layer, deterministic=True))


def interpolate(x, y, n_steps):
    step_size = 1.0/n_steps
    return [x*(1-i) + y*i for i in np.arange(0, 1+step_size, step_size)]

# interpoalte
batch_size = 8
n_steps = 10
fixed_noises = np.random.rand(batch_size, noise_size).astype(np.float32)
noises = [interpolate(fixed_noises[i-1], fixed_noises[i], n_steps)
          for i in range(1, batch_size)]

fig, axes = plt.subplots(1, batch_size-1, figsize=(16, 4))
for i in range(batch_size-1):
    axes[i].imshow(gen_fn(fixed_noises[i].reshape(1, noise_size))[0, 0],
                   aspect='auto', origin='bottom', cmap='gray')
fig.tight_layout()
fig.savefig('images/lsgan_proll/latent_interpolation/z_sources')

fig, axes = plt.subplots(batch_size-1, n_steps+1, figsize=(32, 12))
for i in range(batch_size-1):
    for k in range(n_steps+1):
        axes[i, k].imshow(gen_fn(noises[i][k].reshape(1, noise_size))[0, 0],
                          aspect='auto', origin='bottom', cmap='gray')
fig.tight_layout()
fig.savefig('images/lsgan_proll/latent_interpolation/z_interpolation')

# move all at the same time
