import os
import functools
from collections import defaultdict
import glob2 as glob
import numpy as np
import seaborn as sbn
import theano
import theano.tensor as T
import lasagne
from IPython import display
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import pdb

# load data from each class
datapath = '/Users/rafaelvalle/Desktop/datasets/Piano/'
glob_folder = '/Users/rafaelvalle/Desktop/datasets/Piano/*/'
glob_file = '*.npy'
data_dict = defaultdict(list)
n_pieces = 8
crop = (32, 96)

for folderpath in glob.glob(glob_folder):
    composer = os.path.basename(os.path.normpath(folderpath))
    filepaths = glob.glob(os.path.join(
        os.path.join(datapath, composer), '*.npy'))
    for filepath in np.random.choice(filepaths, n_pieces, replace=False):
        cur_data = np.load(filepath)
        if crop is not None:
            cur_data = cur_data[:, crop[0]:crop[1]]

        # scale each frame [-1, 1]
        cur_data += cur_data.min(axis=1)[:, None]
        cur_data /= cur_data.max(axis=1)[:, None]
        cur_data = np.nan_to_num(cur_data)
        cur_data = cur_data * 2 - 1
        data_dict[composer].append(cur_data)

d_batch_size = g_batch_size = len(data_dict) * 16
min_len = 32
max_len = 64
n_timesteps = 128  # 100ms per step
n_features = data_dict[data_dict.keys()[0]][0].shape[1]
n_conditions = len(data_dict.keys())
temperature = 1.
n_units_d = 8
n_units_g = 32


def output_nonlinearity(data, temperature=1):
    return T.clip(lasagne.nonlinearities.sigmoid(data / temperature),
                  1e-7, 1 - 1e-7)

sigmoid_temperature = functools.partial(
    output_nonlinearity, temperature=temperature)

arch = 1
if arch == 1:
    # declare theano variables
    d_in_X = T.ftensor3('ddata')
    d_in_M = T.imatrix('dismask')
    g_in_D = T.ftensor3('gdata')
    g_in_Z = T.ftensor3('noise')
    g_in_C = T.ftensor3('condition')
    g_in_M = T.imatrix('genmask')

    # declare discriminator specs
    d_specs = {'batch_size': d_batch_size,
               'input_shape': (None, None, n_features),
               'mask_shape': (None, None),
               'n_output_units': 1,
               'n_units': n_units_d,
               'grad_clip': 100.,
               'init': lasagne.init.Orthogonal,
               'non_linearities': (
                   lasagne.nonlinearities.rectify,  # feedforward
                   lasagne.nonlinearities.rectify,  # feedbackward
                   sigmoid_temperature),  # sigmoid with temperature
               'learning_rate': 0.01,
               'regularization': 0.0,
               'unroll': 0,
               'iterations_pre': 100,
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
               'learning_rate': 0.01
               }
elif arch == 2:
    raise Exception("arch 2 not implemented")
elif arch == 3:
    raise Exception("arch 2 not implemented")

# lstm parameters
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


def build_discriminator(params):
    # input layers
    l_in = lasagne.layers.InputLayer(
        shape=params['input_shape'], name='d_in')
    l_mask = lasagne.layers.InputLayer(
        shape=params['mask_shape'], name='d_mask')

    # recurrent layers for bidirectional network
    l_forward = lasagne.layers.LSTMLayer(
        l_in, params['n_units'], grad_clipping=params['grad_clip'],
        # W_in_to_hid=params['init'], W_hid_to_hid=params['init'],
        nonlinearity=params['non_linearities'][0], only_return_final=True,
        mask_input=l_mask)
    l_backward = lasagne.layers.LSTMLayer(
        l_in, params['n_units'], grad_clipping=params['grad_clip'],
        # W_in_to_hid=params['init'], W_hid_to_hid=params['init'],
        nonlinearity=params['non_linearities'][1], only_return_final=True,
        mask_input=l_mask, backwards=True)

    # concatenate output of forward and backward layers
    l_concat = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1)

    # output layer
    l_out = lasagne.layers.DenseLayer(
        l_concat, num_units=params['n_output_units'],
        nonlinearity=params['non_linearities'][2])

    regularization = lasagne.regularization.regularize_layer_params(
        l_out, lasagne.regularization.l2) * params['regularization']

    class Discriminator:
        def __init__(self, l_in, l_mask, l_out):
            self.l_in = l_in
            self.l_mask = l_mask
            self.l_out = l_out
            self.regularization = regularization

    return Discriminator(l_in, l_mask, l_out)


def build_generator_lstm(params, arch=1):
    if arch == 1:
        # input layers
        l_in = lasagne.layers.InputLayer(
            shape=params['input_shape'], input_var=g_in_D, name='g_in')
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
        # l_lstm_dense = lasagne.layer.batch_norm(l_lstm_dense)

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


def sample_data(data, batch_size, min_len, max_len, clip=False):
    encoding = defaultdict(lambda _: 0)
    i = 0
    for k in data.keys():
        encoding[k] = i
        i += 1

    pieces_per_lbl = int(batch_size / len(data.keys()))

    while True:
        inputs, conds, masks = [], [], []
        for k, lbl in encoding.items():
                pieces = np.random.choice(data[k], pieces_per_lbl)
                for piece in pieces:
                    start_idx = np.random.randint(0, piece.shape[0] -max_len -1)
                    piece_data = piece[start_idx: start_idx+max_len]
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


def build_training(discriminator, generator, d_specs, g_specs):
    d_params = lasagne.layers.get_all_params(discriminator.l_out, trainable=True)
    g_params = lasagne.layers.get_all_params(generator.l_out, trainable=True)

    # G(z)
    g_z = lasagne.layers.get_output(generator.l_out,
                                    inputs={generator.l_in: g_in_D,
                                            generator.l_noise: g_in_Z,
                                            generator.l_cond: g_in_C,
                                            generator.l_mask: g_in_M})

    # D(G(z))
    g_z_M = T.ones((g_z.shape[0], g_z.shape[1]))
    d_g_z = lasagne.layers.get_output(discriminator.l_out,
                                      inputs={discriminator.l_in: g_z,
                                              discriminator.l_mask: g_z_M})
    # D(x)
    d_x = lasagne.layers.get_output(discriminator.l_out,
                                    inputs={discriminator.l_in: d_in_X,
                                            discriminator.l_mask: d_in_M})
    # loss functions
    g_loss = lasagne.objectives.binary_crossentropy(
        T.clip(d_g_z, 1e-7, 1.0 - 1e-7), 0.95)
    g_loss = g_loss.mean()

    d_x_loss = lasagne.objectives.binary_crossentropy(
        T.clip(d_x, 1e-7, 1.0 - 1e-7), 0.95)
    d_g_z_loss = lasagne.objectives.binary_crossentropy(
        T.clip(d_g_z, 1e-7, 1.0 - 1e-7), 0.05)

    d_loss = d_x_loss + d_g_z_loss
    d_loss = d_loss.mean()
    d_loss = d_loss + discriminator.regularization

    # update functions
    d_updates = lasagne.updates.sgd(d_loss, d_params, d_specs['learning_rate'])
    if d_specs.get('unroll', 0):
        def fprop(d_in_X, d_in_M, d_labels, g_z):
            return [g_loss, d_updates]

        values, updates = theano.scan(fprop, n_steps=d_specs['unroll'],
                                      non_sequences=[d_in_X, d_in_M, g_z])
        g_loss0 = theano.clone(g_loss, replace=updates)
        g_updates = lasagne.updates.adam(
            g_loss0, g_params, g_specs['learning_rate'])
    else:
        g_updates = lasagne.updates.adam(
            g_loss, g_params, g_specs['learning_rate'])

    # training functions
    d_train_fn = theano.function(
        inputs=[d_in_X, d_in_M, g_in_D, g_in_Z, g_in_C, g_in_M],
        outputs=[d_loss, d_x_loss, d_g_z_loss, d_x, d_g_z, g_z],
        updates=d_updates)
    if d_specs.get('unroll', 0):
        g_train_fn = theano.function(
            inputs=[g_in_D, g_in_Z, g_in_C, g_in_M, d_in_M, d_in_X],
            outputs=g_loss,
            updates=g_updates)
    else:
        g_train_fn = theano.function(
            inputs=[g_in_D, g_in_Z, g_in_C, g_in_M],
            outputs=g_loss,
            updates=g_updates)
    """
    # gradient functions
    grad_tn = theano.grad(
        d_loss, discriminator.l_out.W)
    # grad_tn = theano.grad(
    #     d_loss, lasagne.layers.get_all_params(discriminator.l_out)
    d_grad_fn = theano.function(
        inputs=[d_in_X, d_in_M, g_in_D, g_in_Z, g_in_C, g_in_M],
        outputs=[grad_tn])

    # prediction functions
    d_pred_fn = theano.function(
        inputs=[d_in_X, d_in_M],
        outputs=d_x)
    """

    return d_train_fn, g_train_fn

print("build discriminator")
discriminator = build_discriminator(d_specs)

print("build generator")
generator = build_generator_lstm(g_specs, arch=arch)

print("build training")
d_train_fn, g_train_fn = build_training(
    discriminator, generator, d_specs, g_specs)

# D(x)
data_iter = sample_data(data_dict, d_batch_size, min_len, max_len)

# pre training variables
n_d_iterations_pre = d_specs.get('iterations_pre', 0)
d_losses_pre = []

# training variables
d_losses = []
g_losses = []
n_iterations = int(5e4)
n_d_iterations = 5
n_g_iterations = 1
epoch = 100

folderpath = (
    'piano_multistep_lstm_gan_pre{}'
    'dlr{}glr{}dit{}git{}'
    'bs{}temp{}nunits{}nt{}'
    'crop{}unroll{}').format(n_d_iterations_pre,
                             d_specs['learning_rate'], g_specs['learning_rate'],
                             n_d_iterations, n_g_iterations,
                             d_batch_size, temperature, n_units_g,
                             n_timesteps, crop, d_specs['unroll'])

if not os.path.exists(os.path.join('images', folderpath)):
    os.makedirs(os.path.join('images', folderpath))

# pre training loop
for i in range(d_specs.get('iterations_pre', 0)):
    # use same data for discriminator and generator
    d_X, d_C, d_M = data_iter.next()
    g_X, g_C, g_M = d_X, d_C, d_M

    # gaussian(spherical) noise
    g_Z = np.random.normal(size=g_specs['noise_shape']).astype('float32')

    d_loss, d_x_loss, d_g_z_loss, d_x, d_g_z, g_z = d_train_fn(
        d_X, d_M, g_X, g_Z, g_C, g_M)
    d_losses_pre.append(d_loss)

    if i == (n_d_iterations_pre - 1):
        fig, axes = plt.subplots(4, 2, figsize=(12, 15))
        axes[0, 0].set_title('D(x)')
        sbn.heatmap(d_x.T, ax=axes[0, 0])
        axes[0, 1].set_title('D(G(z))')
        sbn.heatmap(d_g_z.T, ax=axes[0, 1])

        axes[1, 0].set_title('G(z) : 0')
        axes[1, 1].set_title('G(z) : 1')
        axes[2, 0].set_title('G(z) : 2')
        axes[2, 1].set_title('G(z) : 3')
        sbn.heatmap(g_z[0].T, ax=axes[1, 0]).invert_yaxis()
        sbn.heatmap(g_z[1].T, ax=axes[1, 1]).invert_yaxis()
        sbn.heatmap(g_z[2].T, ax=axes[2, 0]).invert_yaxis()
        sbn.heatmap(g_z[3].T, ax=axes[2, 1]).invert_yaxis()
        fig.savefig('images/{}/pretraining'.format(folderpath))
        axes[3, 0].set_title('Loss(d)')
        axes[3, 0].plot(d_losses_pre)
        plt.close('all')

# training loop
for iteration in tqdm(range(n_iterations)):
    for i in range(n_d_iterations):
        # load data
        d_X, d_C, d_M = data_iter.next()
        g_C, g_M = d_C, d_M
        g_Z = np.random.normal(size=g_specs['noise_shape']).astype('float32')

        # randomly add noise to d_X
        if (np.random.random() > 0.66):
            g_X = np.copy(d_X)
            d_X += np.random.normal(0, 1, size=d_X.shape)
        else:
            g_X = d_X

        d_loss, d_x_loss, d_g_z_loss, d_x, d_g_z, g_z = d_train_fn(
            d_X, d_M, g_X, g_Z, g_C, g_M)
        d_losses.append(d_loss)

    for i in range(n_g_iterations):
        # load data
        d_X, d_C, d_M = data_iter.next()
        g_X, g_C, g_M = d_X, d_C, d_M
        g_Z = np.random.normal(size=g_specs['noise_shape']).astype('float32')

        # train iteration
        if d_specs.get('unroll'):
            g_loss = g_train_fn(g_X, g_Z, g_C, g_M, d_M, d_X)
        else:
            g_loss = g_train_fn(g_X, g_Z, g_C, g_M)
        g_losses.append(g_loss)

    if iteration % epoch == 0:
        fig, axes = plt.subplots(6, 2, figsize=(32, 32))
        axes[0, 0].set_title('D(x)')
        sbn.heatmap(d_x.T, ax=axes[0, 0])
        axes[0, 1].set_title('D(G(z))')
        sbn.heatmap(d_g_z.T, ax=axes[0, 1])

        axes[1, 0].set_title('D : 0')
        axes[2, 0].set_title('D : 1')
        axes[3, 0].set_title('D : 2')
        axes[4, 0].set_title('D : 3')
        axes[1, 1].set_title('G(z) : 0')
        axes[2, 1].set_title('G(z) : 1')
        axes[3, 1].set_title('G(z) : 2')
        axes[4, 1].set_title('G(z) : 3')

        sbn.heatmap(g_X[0].T, ax=axes[1, 0]).invert_yaxis()
        sbn.heatmap(g_X[1].T, ax=axes[2, 0]).invert_yaxis()
        sbn.heatmap(g_X[2].T, ax=axes[3, 0]).invert_yaxis()
        sbn.heatmap(g_X[3].T, ax=axes[4, 0]).invert_yaxis()

        sbn.heatmap(g_z[0].T, ax=axes[1, 1]).invert_yaxis()
        sbn.heatmap(g_z[1].T, ax=axes[2, 1]).invert_yaxis()
        sbn.heatmap(g_z[2].T, ax=axes[3, 1]).invert_yaxis()
        sbn.heatmap(g_z[3].T, ax=axes[4, 1]).invert_yaxis()

        axes[5, 0].set_title('Loss(d)')
        axes[5, 0].plot(d_losses)
        axes[5, 1].set_title('Loss(g)')
        axes[5, 1].plot(g_losses)

        fig.savefig('images/{}/iteration_{}'.format(folderpath, iteration))
        plt.close('all')
        display.clear_output(wait=True)
