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
n_pieces = 20
for folderpath in glob.glob(glob_folder):
    composer = os.path.basename(os.path.normpath(folderpath))
    filepaths = glob.glob(os.path.join(
        os.path.join(datapath, composer), '*.npy'))
    for filepath in np.random.choice(filepaths, n_pieces, replace=False):
        data_dict[composer].append(np.load(filepath))


d_batch_size = g_batch_size = len(data_dict) * 16
min_len = 32
max_len = 64
n_timesteps = 32  # 100ms per step
n_features = data_dict[data_dict.keys()[0]][0].shape[1]
n_conditions = len(data_dict.keys())
# n_labels = n_conditions + 1
n_labels = 2
multilabel = False
temperature = 1
n_units_d = 32
n_units_g = 32


def output_nonlinearity(data, temperature=1):
    return T.clip(lasagne.nonlinearities.softmax(lasagne.nonlinearities.linear(data / temperature)), 1e-7, 1 - 1e-7)

softmax_temperature = functools.partial(output_nonlinearity, temperature=temperature)
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
               'n_output_units': n_labels,
               'n_units': 32,
               'grad_clip': 100.,
               'init': lasagne.init.HeUniform(),
               'non_linearities': (
                   lasagne.nonlinearities.tanh,  # feedforward
                   lasagne.nonlinearities.tanh,  # feedbackward
                   softmax_temperature),  # apply sotfmax with temperature
               'learning_rate': 0.01
    }

    # declare generator specs
    g_specs = {'batch_size': g_batch_size,
               'input_shape': (g_batch_size, max_len, n_features),
               'noise_shape': (g_batch_size, max_len, 1),
               'cond_shape': (g_batch_size, max_len, n_conditions),
               'mask_shape': (g_batch_size, max_len),
               'output_shape':  (g_batch_size, n_timesteps, n_features),
               'n_units': [32, 64, n_features],
               'grad_clip': 100.,
               'init': lasagne.init.HeUniform(),
               'non_linearities': (
                   lasagne.nonlinearities.tanh,  # forw and backward data
                   lasagne.nonlinearities.tanh,  # forw and backward noise
                   lasagne.nonlinearities.tanh,  # data and noise lstm concat
                   lasagne.nonlinearities.rectify),
               'learning_rate': 0.01
    }

elif arch == 2:
    d_specs = {'batch_size': d_batch_size,
            'input_shape': (None, None, n_features),
            'mask_shape': (None, None),
            'n_output_units': n_labels,
            'n_units': n_units_d,
            'grad_clip': 100.,
            'init': lasagne.init.HeUniform(),
            'non_linearities': (
                lasagne.nonlinearities.tanh,  # feedforward
                lasagne.nonlinearities.tanh,  # feedbackward
                softmax_temperature),  # apply sotfmax with temperature
            'learning_rate': 0.01
            }

    g_specs = {'batch_size': g_batch_size,
            'input_shape': (g_batch_size, max_len, n_features),
            # 'noise_shape': (g_batch_size, max_len, n_features),
            'noise_shape': (g_batch_size, max_len, 1),
            'cond_shape': (g_batch_size, max_len, n_conditions),
            'mask_shape': (g_batch_size, max_len),
            'output_shape':  (g_batch_size, n_timesteps, n_features),
            'n_output_units': n_features,
            'n_units': n_units_g,
            'grad_clip': 100.,
            'init': lasagne.init.HeUniform(),
            'non_linearities': (
                lasagne.nonlinearities.tanh,  # feedforward
                lasagne.nonlinearities.tanh,  # feedbackward
                lasagne.nonlinearities.rectify),  # apply sotfmax with temperature
            'learning_rate': 0.01
            }
elif arch == 3:
    d_specs = {'batch_size': d_batch_size,
            'input_shape': (None, None, n_features),
            'mask_shape': (None, None),
            'n_output_units': n_labels,
            'n_units': n_units_d,
            'grad_clip': 100.,
            'init': lasagne.init.HeUniform(),
            'non_linearities': (
                lasagne.nonlinearities.tanh,  # feedforward
                lasagne.nonlinearities.tanh,  # feedbackward
                softmax_temperature),  # apply sotfmax with temperature
            'learning_rate': 0.01
            }

    g_specs = {'batch_size': g_batch_size,
            'input_shape': (g_batch_size, max_len, n_features),
            # 'noise_shape': (g_batch_size, max_len, n_features),
            'noise_shape': (g_batch_size, max_len, 1),
            'cond_shape': (g_batch_size, max_len, n_conditions),
            'mask_shape': (g_batch_size, max_len),
            'output_shape':  (g_batch_size, n_timesteps, n_features),
            'n_output_units': n_features,
            'n_units': n_units_g,
            'grad_clip': 100.,
            'init': lasagne.init.HeUniform(),
            'non_linearities': (
                lasagne.nonlinearities.tanh,  # feedforward
                lasagne.nonlinearities.tanh,  # feedbackward
                lasagne.nonlinearities.rectify),  # apply sotfmax with temperature
            'learning_rate': 0.01
            }

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
        shape=d_specs['input_shape'], name='d_in')
    l_mask = lasagne.layers.InputLayer(
        shape=d_specs['mask_shape'], name='d_mask')

    # recurrent layers for bidirectional network
    l_forward = lasagne.layers.RecurrentLayer(
        l_in, d_specs['n_units'], grad_clipping=d_specs['grad_clip'],
        W_in_to_hid=d_specs['init'], W_hid_to_hid=d_specs['init'],
        nonlinearity=d_specs['non_linearities'][0], only_return_final=True,
        mask_input=l_mask)
    l_backward = lasagne.layers.RecurrentLayer(
        l_in, d_specs['n_units'], grad_clipping=d_specs['grad_clip'],
        W_in_to_hid=d_specs['init'], W_hid_to_hid=d_specs['init'],
        nonlinearity=d_specs['non_linearities'][1], only_return_final=True,
        mask_input=l_mask, backwards=True)

    # concatenate output of forward and backward layers
    l_concat = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1)

    # output layer
    l_out = lasagne.layers.DenseLayer(
        l_concat, num_units=d_specs['n_output_units'],
        nonlinearity=d_specs['non_linearities'][2])

    class Discriminator:
        def __init__(self, l_in, l_mask, l_out):
            self.l_in = l_in
            self.l_mask = l_mask
            self.l_out = l_out

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

        # dense layer on output of data and noise lstms
        l_lstm_dense = lasagne.layers.DenseLayer(
            l_lstm_concat, num_units=params['n_units'][1], num_leading_axes=2,
            nonlinearity=params['non_linearities'][2])

        # concatenate dense layer of lstsm with condition
        l_lstm_cond_concat = lasagne.layers.ConcatLayer(
            [l_lstm_dense, l_cond], axis=2)

        # dense layer with dense layer lstm and condition
        l_out = lasagne.layers.DenseLayer(
            l_lstm_cond_concat, num_units=params['n_units'][2],
            num_leading_axes=2,
            nonlinearity=params['non_linearities'][3])

    elif arch == 2:
        # input layers
        l_in = lasagne.layers.InputLayer(
            shape=params['input_shape'], input_var=g_in_D, name='g_in')
        l_noise = lasagne.layers.InputLayer(
            shape=params['noise_shape'], input_var=g_in_Z, name='g_noise')
        l_cond = lasagne.layers.InputLayer(
            shape=params['cond_shape'], input_var=g_in_C, name='g_cond')
        l_mask = lasagne.layers.InputLayer(
            shape=params['mask_shape'], input_var=g_in_M, name='g_mask')
    elif arch == 3:
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
            l_in, params['n_units'], mask_input=l_mask,
            ingate=gate_parameters, forgetgate=gate_parameters,
            cell=cell_parameters, outgate=gate_parameters,
            learn_init=True, grad_clipping=params['grad_clip'],
            only_return_final=False)
        l_forward_noise = lasagne.layers.recurrent.LSTMLayer(
            l_noise, params['n_units'], mask_input=l_mask,
            ingate=gate_parameters, forgetgate=gate_parameters,
            cell=cell_parameters, outgate=gate_parameters,
            learn_init=True, grad_clipping=params['grad_clip'],
            only_return_final=False)
        l_forward_cond = lasagne.layers.recurrent.LSTMLayer(
            l_cond, params['n_units'], mask_input=l_mask,
            ingate=gate_parameters, forgetgate=gate_parameters,
            cell=cell_parameters, outgate=gate_parameters,
            learn_init=True, grad_clipping=params['grad_clip'],
            only_return_final=False)

        l_backward_data = lasagne.layers.recurrent.LSTMLayer(
            l_in, params['n_units'], mask_input=l_mask,
            ingate=gate_parameters, forgetgate=gate_parameters,
            cell=cell_parameters, outgate=gate_parameters,
            learn_init=True, grad_clipping=params['grad_clip'],
            only_return_final=False, backwards=True)
        l_backward_noise = lasagne.layers.recurrent.LSTMLayer(
            l_noise, params['n_units'], mask_input=l_mask,
            ingate=gate_parameters, forgetgate=gate_parameters,
            cell=cell_parameters, outgate=gate_parameters,
            learn_init=True, grad_clipping=params['grad_clip'],
            only_return_final=False, backwards=True)
        l_backward_cond = lasagne.layers.recurrent.LSTMLayer(
            l_cond, params['n_units'], mask_input=l_mask,
            ingate=gate_parameters, forgetgate=gate_parameters,
            cell=cell_parameters, outgate=gate_parameters,
            learn_init=True, grad_clipping=params['grad_clip'],
            only_return_final=False, backwards=True)

        # sum linearities of data and condition on forward and backward recurrent layers
        l_forward_sum = lasagne.layers.ElemwiseSumLayer(
            [l_forward_data, l_forward_noise, l_forward_cond])
        l_backward_sum = lasagne.layers.ElemwiseSumLayer(
            [l_backward_data, l_backward_noise, l_backward_cond])

        # concat layer nonlinearities
        l_concat_input = lasagne.layers.ConcatLayer(
            [l_forward_data, l_forward_noise, l_forward_cond])

        # apply nonlinearity to forward and backward recurrent layers
        l_forward_nonlinearity = lasagne.layers.NonlinearityLayer(l_forward_sum,
            nonlinearity=params['non_linearities'][0])
        l_backward_nonlinearity = lasagne.layers.NonlinearityLayer(l_backward_sum,
            nonlinearity=params['non_linearities'][1])
        # concatenate output of forward and backward layers
        l_concat = lasagne.layers.ConcatLayer(
            [l_forward_nonlinearity, l_backward_nonlinearity], axis=2)
        # l_sum = lasagne.layers.ElemwiseSumLayer([l_forward_nonlinearity, l_backward_nonlinearity])
        l_out = lasagne.layers.DenseLayer(
            l_concat, num_units=params['n_output_units'], num_leading_axes=2)

        """
        # output layer where time is collapsed into one dimension
        l_out_flat = lasagne.layers.DenseLayer(
            l_concat, num_units=params['n_output_units'], nonlinearity=g_specs['non_linearities'][2])

        # reshape to match discriminator's input
        l_out = lasagne.layers.ReshapeLayer(l_out_flat, params['output_shape'])
        """

    class Generator:
        def __init__(self, l_in, l_noise, l_cond, l_mask, l_out):
            self.l_in = l_in
            self.l_noise = l_noise
            self.l_cond = l_cond
            self.l_mask = l_mask
            self.l_out = l_out

    return Generator(l_in, l_noise, l_cond, l_mask, l_out)


def sample_data(data, batch_size, min_len, max_len, clip=False, multilabel=True):
    encoding = defaultdict(lambda _: 0)
    i = 0
    for k in data.keys():
        encoding[k] = i
        i += 1

    pieces_per_lbl = int(batch_size / len(data.keys()))

    while True:
        inputs, targets, labels, conds, masks = [], [], [], [], []
        for k, lbl in encoding.items():
                pieces = np.random.choice(data[k], pieces_per_lbl)
                for piece in pieces:
                    start_idx = np.random.randint(0, piece.shape[0] - max_len - 1)
                    piece_data = piece[start_idx: start_idx+max_len]
                    mask_size = np.random.randint(min_len, max_len)
                    if clip:
                        piece_data[mask_size:] = 0
                    target = piece[start_idx+max_len]
                    if multilabel:
                        label = np.zeros(len(encoding)+1, dtype=np.float32) #+1 for generator label
                        label[encoding[k]] = 1
                    else:
                        label = np.zeros(2, dtype=np.float32)
                        label[0] = 1

                    cond = np.zeros((max_len, len(encoding)), dtype=np.float32) #+1 for generator label
                    cond[:, encoding[k]] = 1

                    mask = np.zeros(max_len, dtype=np.int32)
                    mask[:mask_size] = 1

                    inputs.append(piece_data)
                    targets.append(target)
                    labels.append(label)
                    conds.append(cond)
                    masks.append(mask)

        inputs = np.array(inputs, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)
        conds = np.array(conds, dtype=np.float32)
        masks = np.array(masks, dtype=np.int32)

        shuffle_ids = np.random.randint(0, len(inputs), len(inputs))
        yield inputs[shuffle_ids], targets[shuffle_ids], labels[shuffle_ids], conds[shuffle_ids], masks[shuffle_ids]


def build_training(discriminator, generator, d_specs, g_specs):
    # get variables from discrimiator and generator
    d_params = lasagne.layers.get_all_params(discriminator.l_out, trainable=True)
    g_params = lasagne.layers.get_all_params(generator.l_out, trainable=True)

    d_labels = T.fmatrix('d_label')
    g_labels = T.fmatrix('g_label')

    # G(z)
    g_z = lasagne.layers.get_output(generator.l_out,
                                    inputs={generator.l_in: g_in_D,
                                            generator.l_noise: g_in_Z,
                                            generator.l_cond: g_in_C,
                                            generator.l_mask: g_in_M})

    # create G(z) mask
    g_z_M = T.ones((g_z.shape[0], g_z.shape[1]))

    # D(G(z))
    d_g_z = lasagne.layers.get_output(discriminator.l_out,
                                      inputs={discriminator.l_in: g_z,
                                              discriminator.l_mask: g_z_M})
    #  clip to prevent log(0)
    #d_g_z = T.clip(d_g_z, 1e-7, 1.0 - 1e-7)
    #g_loss = -T.log(d_g_z)
    g_loss = lasagne.objectives.categorical_crossentropy(d_g_z, g_labels)
    g_loss = g_loss.mean()

    g_updates = lasagne.updates.adagrad(g_loss, g_params, g_specs['learning_rate'])
    g_train_fn = theano.function(inputs=[g_in_D, g_in_Z, g_in_C, g_in_M, g_labels],
                                 outputs=g_loss,
                                 updates=g_updates)

    # g_sample_fn = theano.function(inputs=[g_in_D, g_in_Z, g_in_C, g_in_M],
    #                              outputs=g_z)

    # D(x)
    d_x = lasagne.layers.get_output(discriminator.l_out,
                                    inputs={discriminator.l_in: d_in_X,
                                            discriminator.l_mask: d_in_M})
    #  clip to prevent log(0)
    # d_x = T.clip(d_x, 1e-7, 1.0 - 1e-7)
    # d_x_loss = -T.log(d_x)
    # d_g_z_loss = -T.log(1 - d_g_z)

    d_x_loss = lasagne.objectives.categorical_crossentropy(d_x, d_labels)
    d_g_z_loss = lasagne.objectives.categorical_crossentropy(1 - d_g_z, g_labels)
    d_loss = d_x_loss + d_g_z_loss
    d_loss = d_loss.mean()

    d_updates = lasagne.updates.adagrad(d_loss, d_params, d_specs['learning_rate'])
    d_train_fn = theano.function(
        inputs=[d_in_X, d_in_M, g_in_D, g_in_Z, g_in_C, g_in_M, d_labels, g_labels],
        outputs=[d_loss, d_x_loss, d_g_z_loss, d_x, d_g_z, g_z],
        updates=d_updates)

    d_predict_fn = None
    g_sample_fn = None
#   d_predict_fn = theano.function(
#        inputs=[d_in_X, d_in_M],
#        outputs=lasagne.layers.get_output(discriminator[2],
#                                          inputs={discriminator[0]: d_in_X,
#                                                  discriminator[1]: d_in_M}))

    return d_train_fn, d_predict_fn, g_train_fn, g_sample_fn


discriminator = build_discriminator(d_specs)
generator = build_generator_lstm(g_specs)

d_train_fn, d_predict_fn, g_train_fn, g_sample_fn = build_training(
    discriminator, generator, d_specs, g_specs)

data_iter = sample_data(
    data_dict, d_batch_size, min_len, max_len, multilabel=multilabel)

## Pre training
n_d_iterations_pre = 1
d_losses_pre = []

folderpath = 'piano_multistep_lstm_gan_pre_{}_dlr_{}_glr_{}_bs_{}_temp_{}_nunits_{}_nt{}_lbls_{}'.format(
    n_d_iterations_pre, d_specs['learning_rate'], g_specs['learning_rate'], d_batch_size, temperature, n_units_g, n_timesteps, n_labels)

if not os.path.exists(os.path.join('images', folderpath)):
    os.makedirs(os.path.join('images', folderpath))

for i in range(n_d_iterations_pre):
    # use same data for discriminator and generator
    d_X, _, d_L, d_C, d_M = data_iter.next()
    g_Z = np.random.normal(size=g_specs['noise_shape']).astype('float32')
    g_L = np.zeros((g_Z.shape[0], n_labels), dtype=np.float32)
    g_L[:, -1] = 1
    g_D, g_C, g_M = d_X, d_C, d_M
    d_loss, d_x_loss, d_g_z_loss, d_x, d_g_z, g_z = d_train_fn(d_X, d_M, g_D, g_Z, g_C, g_M, d_L, g_L)
    d_losses_pre.append(d_loss)

    if i == (n_d_iterations_pre -1):
        fig, axes = plt.subplots(5, 2, figsize=(16, 20))
        axes[0, 0].set_title('D Label')
        sbn.heatmap(d_L.T, ax=axes[0, 0])
        axes[0, 1].set_title('G Label')
        sbn.heatmap(g_L.T, ax=axes[0, 1])

        axes[1, 0].set_title('D(x)')
        sbn.heatmap(d_x.T, ax=axes[1, 0])
        axes[1, 1].set_title('D(G(z))')
        sbn.heatmap(d_g_z.T, ax=axes[1, 1])

        axes[2, 0].set_title('G(z) : 0')
        axes[2, 1].set_title('G(z) : 1')
        axes[3, 0].set_title('G(z) : 2')
        axes[3, 1].set_title('G(z) : 3')
        sbn.heatmap(g_z[0].T, ax=axes[2, 0]).invert_yaxis()
        sbn.heatmap(g_z[1].T, ax=axes[2, 1]).invert_yaxis()
        sbn.heatmap(g_z[2].T, ax=axes[3, 0]).invert_yaxis()
        sbn.heatmap(g_z[3].T, ax=axes[3, 1]).invert_yaxis()
        fig.savefig('images/{}/pretraining'.format(folderpath))
        axes[4, 0].set_title('Loss(d)')
        axes[4, 0].plot(d_losses_pre)
        plt.close('all')


## store losses
d_losses = []
g_losses = []

n_iterations = int(1e4)
n_d_iterations = 1
n_g_iterations = 1
epoch = 100

for iteration in tqdm(range(n_iterations)):
    for i in range(n_d_iterations):
        d_X, d_T, d_L, d_C, d_M = data_iter.next()
        g_Z = np.random.normal(size=g_specs['noise_shape']).astype('float32')
        g_L = np.zeros((g_Z.shape[0], n_labels), dtype=np.float32)
        g_L[:, -1] = 1
        g_D, g_C, g_M = d_X, d_C, d_M
        d_loss, d_x_loss, d_g_z_loss, d_x, d_g_z, g_z = d_train_fn(d_X, d_M, g_D, g_Z, g_C, g_M, d_L, g_L)
        d_losses.append(d_loss)

    for i in range(n_g_iterations):
        d_X, d_T, d_L, d_C, d_M = data_iter.next()
        g_Z = np.random.normal(size=g_specs['noise_shape']).astype('float32')
        g_D, g_C, g_M = d_X, d_C, d_M
        g_loss = g_train_fn(g_D, g_Z, g_C, g_M, g_L)
        g_losses.append(g_loss)

    if iteration % epoch == 0:
        fig, axes = plt.subplots(7, 2, figsize=(32, 32))
        axes[0, 0].set_title('D Label')
        sbn.heatmap(d_L.T, ax=axes[0, 0])
        axes[0, 1].set_title('G Label')
        sbn.heatmap(g_L.T, ax=axes[0, 1])

        axes[1, 0].set_title('D(x)')
        sbn.heatmap(d_x.T, ax=axes[1, 0])
        axes[1, 1].set_title('D(G(z))')
        sbn.heatmap(d_g_z.T, ax=axes[1, 1])

        axes[2, 0].set_title('D : 0')
        axes[3, 0].set_title('D : 1')
        axes[4, 0].set_title('D : 2')
        axes[5, 0].set_title('D : 3')
        axes[2, 1].set_title('G(z) : 0')
        axes[3, 1].set_title('G(z) : 1')
        axes[4, 1].set_title('G(z) : 2')
        axes[5, 1].set_title('G(z) : 3')

        sbn.heatmap(g_D[0].T, ax=axes[2, 0]).invert_yaxis()
        sbn.heatmap(g_D[1].T, ax=axes[3, 0]).invert_yaxis()
        sbn.heatmap(g_D[2].T, ax=axes[4, 0]).invert_yaxis()
        sbn.heatmap(g_D[3].T, ax=axes[5, 0]).invert_yaxis()

        sbn.heatmap(g_z[0].T, ax=axes[2, 1]).invert_yaxis()
        sbn.heatmap(g_z[1].T, ax=axes[3, 1]).invert_yaxis()
        sbn.heatmap(g_z[2].T, ax=axes[4, 1]).invert_yaxis()
        sbn.heatmap(g_z[3].T, ax=axes[5, 1]).invert_yaxis()

        axes[6, 0].set_title('Loss(d)')
        axes[6, 0].plot(d_losses)
        axes[6, 1].set_title('Loss(g)')
        axes[6, 1].plot(g_losses)

        fig.savefig('images/{}/iteration_{}'.format(folderpath, iteration))
        plt.close('all')
        display.clear_output(wait=True)
