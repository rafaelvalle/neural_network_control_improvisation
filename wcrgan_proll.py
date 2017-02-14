import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['axes.xmargin'] = 0.1
matplotlib.rcParams['axes.ymargin'] = 0.1
import matplotlib.pylab as plt
plt.style.use('ggplot')
plt.ioff()
from IPython import display
from tqdm import tqdm

import os
from collections import defaultdict

import numpy as np
import theano
import theano.tensor as T
import lasagne

from data_processing import load_data
import pdb

# dataset params and load data
datapath = '/media/steampunkhd/rafaelvalle/datasets/MIDI/Piano'
glob_file_str = '*.npy'
n_pieces = 1  # 0 is equal to all pieces, unbalanced dataset
crop = None  # (32, 96)
as_dict = True
dataset = load_data(datapath, glob_file_str, n_pieces, crop, as_dict)

# model params
c_batch_size = g_batch_size = 512
n_timesteps = 100  # 100 ms per step
min_len = 50
max_len = 100
single_len = True  # single length per mini-batch
n_features = dataset[dataset.keys()[0]][0].shape[1]
n_conditions = len(dataset.keys())
n_units_d = 8
n_units_g = 16

# declare theano variables
c_in_X = T.ftensor3('ddata')
c_in_M = T.imatrix('dismask')
g_in_X = T.ftensor3('gdata')
g_in_Z = T.ftensor3('noise')
g_in_C = T.ftensor3('condition')
g_in_M = T.imatrix('genmask')

# declare critic specs
c_specs = {'batch_size': c_batch_size,
           'input_shape': (None, None, n_features),
           'mask_shape': (None, None),
           'n_output_units': 1,
           'n_units': n_units_d,
           'grad_clip': 100.,
           'init': lasagne.init.Orthogonal,
           'non_linearities': (
               lasagne.nonlinearities.rectify,  # feedforward
               lasagne.nonlinearities.rectify,  # feedbackward
               None),  # Wassertstein critic has linear output
           'learning_rate': 1e-3,
           'regularization': 0.0,
           }

# declare generator specs
g_specs = {'batch_size': g_batch_size,
           'input_shape': (g_batch_size, max_len, n_features),
           'noise_shape': (g_batch_size, max_len, int(np.sqrt(n_features)*2)),
           'cond_shape': (g_batch_size, max_len, n_conditions),
           'mask_shape': (g_batch_size, max_len),
           'output_shape':  (g_batch_size, n_timesteps, n_features),
           'n_units': [32, 64, n_features],
           'grad_clip': 100.,
           'init': lasagne.init.Orthogonal(),
           'non_linearities': (
               lasagne.nonlinearities.rectify,  # forw and backward data
               lasagne.nonlinearities.rectify,  # forw and backward noise
               lasagne.nonlinearities.rectify,  # data and noise lstm concat
               lasagne.nonlinearities.tanh),
           'learning_rate': 1e-3
           }

# lstm gate and cell parameters
gate_parameters = lasagne.layers.recurrent.Gate(
    W_in=lasagne.init.Orthogonal(),
    W_hid=lasagne.init.Orthogonal(),
    b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.sigmoid)

cell_parameters = lasagne.layers.recurrent.Gate(
    W_in=lasagne.init.Orthogonal(),
    W_hid=lasagne.init.Orthogonal(),
    W_cell=None, b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.tanh)


def build_critic(params):
    # input layers
    l_in = lasagne.layers.InputLayer(
        shape=params['input_shape'], name='c_in')
    l_mask = lasagne.layers.InputLayer(
        shape=params['mask_shape'], name='c_mask')

    # recurrent layers for bidirectional network
    l_forward = lasagne.layers.LSTMLayer(
        l_in, params['n_units'], grad_clipping=params['grad_clip'],
        ingate=gate_parameters, forgetgate=gate_parameters,
        cell=cell_parameters, outgate=gate_parameters,
        nonlinearity=params['non_linearities'][0], only_return_final=True,
        mask_input=l_mask)
    l_backward = lasagne.layers.LSTMLayer(
        l_in, params['n_units'], grad_clipping=params['grad_clip'],
        ingate=gate_parameters, forgetgate=gate_parameters,
        cell=cell_parameters, outgate=gate_parameters,
        nonlinearity=params['non_linearities'][1], only_return_final=True,
        mask_input=l_mask, backwards=True)

    # concatenate output of forward and backward layers
    l_concat = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1)

    # fully-connected layer
    l_full = lasagne.layers.batch_norm(
        lasagne.layers.DenseLayer(l_concat, 1024,
                                  nonlinearity=lasagne.nonlinearities.rectify))

    # output layer
    l_out = lasagne.layers.DenseLayer(
        l_full, num_units=params['n_output_units'],
        nonlinearity=params['non_linearities'][2], b=None)

    regularization = lasagne.regularization.regularize_layer_params(
        l_out, lasagne.regularization.l2) * params['regularization']

    class critic:
        def __init__(self, l_in, l_mask, l_out):
            self.l_in = l_in
            self.l_mask = l_mask
            self.l_out = l_out
            self.regularization = regularization

    return critic(l_in, l_mask, l_out)


def build_generator_lstm(params):
    # input layers
    l_in = lasagne.layers.InputLayer(
        shape=params['input_shape'], input_var=g_in_X, name='g_in')
    l_noise = lasagne.layers.InputLayer(
        shape=params['noise_shape'], input_var=g_in_Z, name='g_noise')
    l_cond = lasagne.layers.InputLayer(
        shape=params['cond_shape'], input_var=g_in_C, name='g_cond')
    l_mask = lasagne.layers.InputLayer(
        shape=params['mask_shape'], input_var=g_in_M, name='g_mask')

    # recurrent layers for bidirectional network
    l_forward_data = lasagne.layers.recurrent.LSTMLayer(
        l_in, params['n_units'][0], mask_input=l_mask,
        ingate=gate_parameters, forgetgate=gate_parameters,
        cell=cell_parameters, outgate=gate_parameters,
        learn_init=True, grad_clipping=params['grad_clip'],
        only_return_final=False,
        nonlinearity=params['non_linearities'][0])
    l_forward_noise = lasagne.layers.recurrent.LSTMLayer(
        l_noise, params['n_units'][0], mask_input=l_mask,
        ingate=gate_parameters, forgetgate=gate_parameters,
        cell=cell_parameters, outgate=gate_parameters,
        learn_init=True, grad_clipping=params['grad_clip'],
        only_return_final=False,
        nonlinearity=params['non_linearities'][1])

    l_backward_data = lasagne.layers.recurrent.LSTMLayer(
        l_in, params['n_units'][0], mask_input=l_mask,
        ingate=gate_parameters, forgetgate=gate_parameters,
        cell=cell_parameters, outgate=gate_parameters,
        learn_init=True, grad_clipping=params['grad_clip'],
        only_return_final=False, backwards=True,
        nonlinearity=params['non_linearities'][0])
    l_backward_noise = lasagne.layers.recurrent.LSTMLayer(
        l_noise, params['n_units'][0], mask_input=l_mask,
        ingate=gate_parameters, forgetgate=gate_parameters,
        cell=cell_parameters, outgate=gate_parameters,
        learn_init=True, grad_clipping=params['grad_clip'],
        only_return_final=False, backwards=True,
        nonlinearity=params['non_linearities'][1])

    # concatenate output of forward and backward layers
    l_lstm_concat = lasagne.layers.ConcatLayer(
        [l_forward_data, l_forward_noise, l_backward_data,
            l_backward_noise], axis=2)

    # dense layer on output of data and noise lstms, w/dropout
    l_lstm_dense = lasagne.layers.DenseLayer(
        lasagne.layers.DropoutLayer(l_lstm_concat, p=0.5),
        num_units=params['n_units'][1], num_leading_axes=2,
        W=lasagne.init.HeNormal(gain='relu'), b=lasagne.init.Constant(0.1),
        nonlinearity=params['non_linearities'][2])

    # batch norm for lstm dense
    l_lstm_dense = lasagne.layers.batch_norm(l_lstm_dense)

    # concatenate dense layer of lstsm with condition
    l_lstm_cond_concat = lasagne.layers.ConcatLayer(
        [l_lstm_dense, l_cond], axis=2)

    # dense layer with dense layer lstm and condition, w/dropout
    l_out = lasagne.layers.DenseLayer(
        lasagne.layers.DropoutLayer(l_lstm_cond_concat, p=0.5),
        num_units=params['n_units'][2],
        num_leading_axes=2,
        W=lasagne.init.HeNormal(gain=1.0), b=lasagne.init.Constant(0.1),
        nonlinearity=params['non_linearities'][3])
    l_out = lasagne.layers.batch_norm(l_out)

    class Generator:
        def __init__(self, l_in, l_noise, l_cond, l_mask, l_out):
            self.l_in = l_in
            self.l_noise = l_noise
            self.l_cond = l_cond
            self.l_mask = l_mask
            self.l_out = l_out

    return Generator(l_in, l_noise, l_cond, l_mask, l_out)


def sample_data(data, batch_size, min_len, max_len, clip=False,
                single_len=True):
    encoding = defaultdict(lambda _: 0)
    i = 0
    for k in data.keys():
        encoding[k] = i
        i += 1

    while True:
        inputs, conds, masks = [], [], []
        if single_len:
            # same length within batch
            mask_size = np.random.randint(min_len, max_len)

        for composer in np.random.choice(data.keys(), batch_size):
            piece = np.random.choice(data[composer])
            start_idx = np.random.randint(0, piece.shape[0] - max_len - 1)
            piece_data = piece[start_idx: start_idx+max_len]
            if not single_len:
                # different lengths within batch
                mask_size = np.random.randint(min_len, max_len)
            if clip:
                piece_data[mask_size:] = 0

            cond = np.zeros((max_len, len(encoding)), dtype=np.float32)
            cond[:, encoding[k]] = 1

            mask = np.zeros(max_len, dtype=np.int32)
            mask[:mask_size] = 1

            inputs.append(piece_data)
            conds.append(cond)
            masks.append(mask)

        shuffle_ids = np.random.randint(0, len(inputs), len(inputs))
        inputs = np.array(inputs, dtype=np.float32)[shuffle_ids]
        conds = np.array(conds, dtype=np.float32)[shuffle_ids]
        masks = np.array(masks, dtype=np.int32)[shuffle_ids]

        yield inputs, conds, masks


def builc_training(critic, generator, c_specs, g_specs, add_noise=True,
                   clip=0.01):
    # Instantiate a symbolic noise generator to use for training
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    srng = RandomStreams(seed=np.random.randint(2147462579, size=6))
    noise = srng.normal(size=g_specs['noise_shape'], avg=0.0, std=1.0)

    # G(z)
    g_z = lasagne.layers.get_output(generator.l_out,
                                    inputs={generator.l_in: g_in_X,
                                            generator.l_noise: noise,
                                            generator.l_cond: g_in_C,
                                            generator.l_mask: g_in_M})

    # C(G(z))
    g_z_M = T.ones((g_z.shape[0], g_z.shape[1]))  # mask
    c_g_z = lasagne.layers.get_output(critic.l_out,
                                      inputs={critic.l_in: g_z,
                                              critic.l_mask: g_z_M})
    # C(x)
    c_x = lasagne.layers.get_output(critic.l_out,
                                    inputs={critic.l_in: c_in_X,
                                            critic.l_mask: c_in_M})
    # loss functions
    g_loss = c_g_z.mean()
    c_loss = c_x.mean() - c_g_z.mean()
    c_loss = c_loss + critic.regularization

    # model params
    g_params = lasagne.layers.get_all_params(generator.l_out, trainable=True)
    c_params = lasagne.layers.get_all_params(critic.l_out, trainable=True)

    # update functions
    g_updates = lasagne.updates.rmsprop(
        -g_loss, g_params, learning_rate=c_specs['learning_rate'])
    c_updates = lasagne.updates.rmsprop(
        -c_loss, c_params, learning_rate=g_specs['learning_rate'])

    # Clip critic parameters in a limited range around zero (except biases)
    for param in lasagne.layers.get_all_params(critic.l_out,
                                               trainable=True,
                                               regularizable=True):
        c_updates[param] = T.clip(c_updates[param], -clip, clip)

    # training functions
    g_train_fn = theano.function(
        inputs=[g_in_X, g_in_C, g_in_M],
        outputs=g_loss,
        updates=g_updates,
        name='g_train')

    c_train_fn = theano.function(
        inputs=[c_in_X, c_in_M, g_in_X, g_in_C, g_in_M],
        outputs=c_loss,
        updates=c_updates,
        name='c_train')

    """
    # gradient functions
    grad_tn = theano.grad(
        c_loss, critic.l_out.W)
    # grad_tn = theano.grad(
    #     c_loss, lasagne.layers.get_all_params(critic.l_out)
    c_grad_fn = theano.function(
        inputs=[c_in_X, c_in_M, g_in_X, g_in_Z, g_in_C, g_in_M],
        outputs=[grad_tn])

    # prediction functions
    c_pred_fn = theano.function(
        inputs=[c_in_X, c_in_M],
        outputs=c_x)
    """

    # sampling function
    g_sample_fn = theano.function(
        inputs=[g_in_X, g_in_Z, g_in_C, g_in_M],
        outputs=lasagne.layers.get_output(generator.l_out,
                                          inputs={generator.l_in: g_in_X,
                                                  generator.l_noise: g_in_Z,
                                                  generator.l_cond: g_in_C,
                                                  generator.l_mask: g_in_M}))
    return c_train_fn, g_train_fn, g_sample_fn

print("Build critic")
critic = build_critic(c_specs)

print("Build generator")
generator = build_generator_lstm(g_specs)

print("Build training")
c_train_fn, g_train_fn, g_sample_fn = builc_training(
    critic, generator, c_specs, g_specs)

print("Create data iterator")
data_iter = sample_data(dataset, c_batch_size, min_len, max_len,
                        single_len=single_len)

# training and pre-training variables
n_epochs = 1000
samples_per_composer = 10 * 180  # 10 pieces, 180 sampes /piece
epoch_size = int(len(dataset) * samples_per_composer / c_batch_size)

folderpath = (
    'wcrgan_'
    'dlr{}glr{}'
    'bs{}nunits{}nt{}'
    'crop{}_esize{}').format(
        c_specs['learning_rate'], g_specs['learning_rate'],
        c_batch_size, n_units_g,
        n_timesteps, crop, epoch_size)

if not os.path.exists(os.path.join('images', folderpath)):
    os.makedirs(os.path.join('images', folderpath))

# training loop
print("Training {} epochs, {} mini-batches each, mini-batch size {}".format(
    n_epochs, epoch_size, c_batch_size))

generator_updates = 0
c_epoch_losses = []
g_epoch_losses = []
for epoch in tqdm(range(n_epochs)):
    # In each epoch, we do `epochsize` generator updates. Usually, the
    # critic is updated 5 times before every generator update. For the
    # first 25 generator updates and every 500 generator updates, the
    # critic is updated 100 times instead, following the authors' code.
    for _ in range(epoch_size):
        c_losses = []
        g_losses = []
        if (generator_updates < 25) or (generator_updates % 500 == 0):
            critic_runs = 100
        else:
            critic_runs = 5
        for _ in range(critic_runs):
            # load mini-batch
            c_X, c_C, c_M = data_iter.next()
            g_C, g_M = c_C, c_M

            # randomly add noise to c_X
            #if (np.random.random() > 0.66):
            #    g_X = np.copy(c_X)
            #    c_X += np.random.normal(0, 1, size=c_X.shape)
            #else:
            g_X = c_X

            c_losses.append(c_train_fn(c_X, c_M, g_X, g_C, g_M))

        # load mini-batch and train generator
        c_X, c_C, c_M = data_iter.next()
        g_X, g_C, g_M = c_X, c_C, c_M
        g_loss = g_train_fn(g_X, g_C, g_M)
        generator_updates += 1

        c_epoch_losses.append(np.mean(c_losses))
        g_epoch_losses.append(g_loss)

        fig, axes = plt.subplots(1, 3, figsize=(8, 2))
        axes[0].set_title('Loss(d)')
        axes[0].plot(c_losses)
        axes[1].set_title('Mean(Loss(d))')
        axes[1].plot(c_epoch_losses)
        axes[2].set_title('Loss(g)')
        axes[2].plot(g_epoch_losses)
        fig.tight_layout()
        fig.savefig('images/{}/g_updates{}'.format(folderpath, generator_updates))

    noise = lasagne.utils.floatX(np.random.normal(size=g_specs['noise_shape']))
    rand_ids = np.random.randint(0, g_specs['noise_shape'][0], 64)
    samples = g_sample_fn(c_X, noise, c_C, c_M)[rand_ids]
    plt.imsave('images/{}/epoch{}_samples.png'.format(folderpath, epoch),
               (samples.reshape(8, 8, max_len, n_features)
                       .transpose(0, 2, 1, 3)
                       .reshape(8*max_len, 8*n_features)).T,
               cmap='gray',
               origin='bottom')

    plt.close('all')
    display.clear_output(wait=True)