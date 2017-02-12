import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['axes.xmargin'] = 0.1
matplotlib.rcParams['axes.ymargin'] = 0.1
import matplotlib.pylab as plt
plt.style.use('ggplot')
plt.ioff()
import os
import functools
from collections import defaultdict
import numpy as np
import theano
import theano.tensor as T
import lasagne
from IPython import display
from tqdm import tqdm
from data_processing import load_data
import pdb

# data params
datapath = '/media/steampunkhd/rafaelvalle/datasets/MIDI/Piano'
glob_file_str = '*.npy'
n_pieces = 8  # 0 is equal to all pieces, unbalanced dataset
crop = None  # (32, 96)
as_dict = True

# load data, takes time depending on dataset site
dataset = load_data(datapath, glob_file_str, n_pieces, crop, as_dict)

# model params
d_batch_size = g_batch_size = 512
n_timesteps = 100  # 100 ms per step
min_len = 50
max_len = 100
single_len = True
n_features = dataset[dataset.keys()[0]][0].shape[1]
n_conditions = len(dataset.keys())
temperature = 1.
n_units_d = 8
n_units_g = 16
arch = 1

# Minibatch Layer from
# https://github.com/openai/improved-gan/blob/master/mnist_svhn_cifar10/nn.py
def output_nonlinearity(x, temperature=1):
    return T.clip(lasagne.nonlinearities.sigmoid(x / temperature),
                  1e-7, 1 - 1e-7)

sigmoid_temperature = functools.partial(
    output_nonlinearity, temperature=temperature)

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
               'learning_rate': 2e-5,
               'regularization': 1e-5,
               'unroll': 0,
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
               'learning_rate': 2e-5
               }
elif arch == 2:
    raise Exception("arch 2 not implemented")
elif arch == 3:
    raise Exception("arch 2 not implemented")

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


# minibatch discrimination layer
# https://github.com/openai/improved-gan/blob/master/mnist_svhn_cifar10/nn.py
class MinibatchLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_kernels, dim_per_kernel=5, theta=lasagne.init.Normal(0.05),
                 log_weight_scale=lasagne.init.Constant(0.), b=lasagne.init.Constant(-1.), **kwargs):
        super(MinibatchLayer, self).__init__(incoming, **kwargs)
        self.num_kernels = num_kernels
        num_inputs = int(np.prod(self.input_shape[1:]))
        self.theta = self.add_param(theta, (num_inputs, num_kernels, dim_per_kernel), name="theta")
        self.log_weight_scale = self.add_param(log_weight_scale, (num_kernels, dim_per_kernel), name="log_weight_scale")
        self.W = self.theta * (T.exp(self.log_weight_scale)/T.sqrt(T.sum(T.square(self.theta),axis=0))).dimshuffle('x',0,1)
        self.b = self.add_param(b, (num_kernels,), name="b")
        
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], np.prod(input_shape[1:])+self.num_kernels)

    def get_output_for(self, input, init=False, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)
        
        activation = T.tensordot(input, self.W, [[1], [0]])
        abs_dif = (T.sum(abs(activation.dimshuffle(0,1,2,'x') - activation.dimshuffle('x',1,2,0)),axis=2)
                    + 1e6 * T.eye(input.shape[0]).dimshuffle(0,'x',1))

        if init:
            mean_min_abs_dif = 0.5 * T.mean(T.min(abs_dif, axis=2),axis=0)
            abs_dif /= mean_min_abs_dif.dimshuffle('x',0,'x')
            self.init_updates = [(self.log_weight_scale, self.log_weight_scale-T.log(mean_min_abs_dif).dimshuffle(0,'x'))]
        
        f = T.sum(T.exp(-abs_dif),axis=2)

        if init:
            mf = T.mean(f,axis=0)
            f -= mf.dimshuffle('x',0)
            self.init_updates.append((self.b, -mf))
        else:
            f += self.b.dimshuffle('x',0)

        return T.concatenate([input, f], axis=1)


def build_discriminator(params):
    # input layers
    l_in = lasagne.layers.InputLayer(
        shape=params['input_shape'], name='d_in')
    l_mask = lasagne.layers.InputLayer(
        shape=params['mask_shape'], name='d_mask')

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

    # minibatch layer on forward and backward layers
    l_minibatch = MinibatchLayer(l_concat, num_kernels=100)

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


def build_training(discriminator, generator, d_specs, g_specs, add_noise=True):
    # Instantiate a symbolic noise generator to use for training
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    srng = RandomStreams(seed=np.random.randint(2147462579, size=6))
    noise = srng.normal(size=g_specs['noise_shape'], avg=0.0, std=1.0)

    # one-sided label smoothing
    lbl_noise = 0.0
    if add_noise:
        print("  adding label noise")
        lbl_noise = srng.normal(size=(1,), avg=0.0, std=0.1)

    # model params
    d_params = lasagne.layers.get_all_params(discriminator.l_out, trainable=True)
    g_params = lasagne.layers.get_all_params(generator.l_out, trainable=True)

    # G(z)
    g_z = lasagne.layers.get_output(generator.l_out,
                                    inputs={generator.l_in: g_in_D,
                                            generator.l_noise: noise,
                                            generator.l_cond: g_in_C,
                                            generator.l_mask: g_in_M})

    # D(G(z))
    g_z_M = T.ones((g_z.shape[0], g_z.shape[1]))  # mask
    d_g_z = lasagne.layers.get_output(discriminator.l_out,
                                      inputs={discriminator.l_in: g_z,
                                              discriminator.l_mask: g_z_M})
    # D(x)
    d_x = lasagne.layers.get_output(discriminator.l_out,
                                    inputs={discriminator.l_in: d_in_X,
                                            discriminator.l_mask: d_in_M})
    # loss functions
    g_loss = lasagne.objectives.binary_crossentropy(
        T.clip(d_g_z, 1e-7, 1.0 - 1e-7), 1.0)
    g_loss = g_loss.mean()
    d_x_loss = lasagne.objectives.binary_crossentropy(
        d_x, 1.0 + lbl_noise)
    d_g_z_loss = lasagne.objectives.binary_crossentropy(
        d_g_z, 0.0)

    d_loss = d_x_loss + d_g_z_loss
    d_loss = d_loss.mean()
    d_loss = d_loss + discriminator.regularization

    # update functions
    d_updates = lasagne.updates.sgd(d_loss, d_params, d_specs['learning_rate'])
    if d_specs.get('unroll', 0):
        def fprop(d_in_X, d_in_M, g_z):
            return [g_loss, d_updates]

        values, updates = theano.scan(fprop, n_steps=d_specs['unroll'],
                                      non_sequences=[d_in_X, d_in_M, g_z])
        g_loss0 = theano.clone(g_loss, replace=updates)
        g_updates = lasagne.updates.adam(
            g_loss0, g_params, g_specs['learning_rate'], beta1=0.5)
    else:
        g_updates = lasagne.updates.adam(
            g_loss, g_params, g_specs['learning_rate'], beta1=0.5)

    # objective function
    acc_d_x = (d_x > .5).mean()
    acc_d_g_z = (d_g_z < .5).mean()

    # training functions
    d_train_fn = theano.function(
        inputs=[d_in_X, d_in_M, g_in_D, g_in_C, g_in_M],
        outputs=[d_loss, acc_d_x, acc_d_g_z],
        updates=d_updates,
        name='d_train')
    if d_specs.get('unroll', 0):
        g_train_fn = theano.function(
            inputs=[g_in_D, g_in_C, g_in_M, d_in_M, d_in_X],
            outputs=g_loss,
            updates=g_updates,
            name='g_train_unroll')
    else:
        g_train_fn = theano.function(
            inputs=[g_in_D, g_in_C, g_in_M],
            outputs=g_loss,
            updates=g_updates,
            name='g_train')
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

    # accuracy functions
    d_acc_fn = theano.function(
        inputs=[d_in_X, d_in_M],
        outputs=acc_d_x)

    # sampling function
    g_sample_fn = theano.function(
        inputs=[g_in_D, g_in_Z, g_in_C, g_in_M],
        outputs=lasagne.layers.get_output(generator.l_out,
                                          inputs={generator.l_in: g_in_D,
                                                  generator.l_noise: g_in_Z,
                                                  generator.l_cond: g_in_C,
                                                  generator.l_mask: g_in_M}))
    return d_train_fn, g_train_fn, g_sample_fn, d_acc_fn

print("Build discriminator")
discriminator = build_discriminator(d_specs)

print("Build generator")
generator = build_generator_lstm(g_specs, arch=arch)

print("Build training")
d_train_fn, g_train_fn, g_sample_fn, d_acc_fn = build_training(
    discriminator, generator, d_specs, g_specs)

print("Create data iterator")
data_iter = sample_data(dataset, d_batch_size, min_len, max_len,
                        single_len=single_len)

# training and pre-training variables
n_d_iterations_pre = 5
n_epochs = 1000
samples_per_composer = 10 * 1800  # 10 pieces, 1800 frames/piece
epoch_size = int(len(dataset) * samples_per_composer / d_batch_size)
n_d_iterations = 1
n_g_iterations = 1

folderpath = (
    'piano_multistep_lstm_gan_pre{}'
    'dlr{}glr{}dit{}git{}'
    'bs{}temp{}nunits{}nt{}'
    'crop{}unroll{}_esize{}').format(
        n_d_iterations_pre, d_specs['learning_rate'], g_specs['learning_rate'],
        n_d_iterations, n_g_iterations, d_batch_size, temperature, n_units_g,
        n_timesteps, crop, d_specs['unroll'], epoch_size)

if not os.path.exists(os.path.join('images', folderpath)):
    os.makedirs(os.path.join('images', folderpath))

# pre training loop
d_losses = []
d_acc_x = []
d_acc_g_z = []
print("Pre-training {} iterations".format(n_d_iterations_pre))
for i in range(n_d_iterations_pre):
    # use same data for discriminator and generator
    d_X, d_C, d_M = data_iter.next()
    g_X, g_C, g_M = d_X, d_C, d_M

    d_loss, acc_d_x, acc_d_g_z = d_train_fn(d_X, d_M, g_X, g_C, g_M)
    d_losses.append(d_loss)
    d_acc_x.append(acc_d_x)
    d_acc_g_z.append(acc_d_g_z)

    if i == (n_d_iterations_pre - 1):
        fig, axes = plt.subplots(1, 2, figsize=(4, 3))
        axes[0].set_title('Loss(d)')
        axes[0].plot(d_losses)
        axes[1].set_title('Accuracy D(x) and D(G(z))')
        axes[1].plot(d_acc_x, color='blue')
        axes[1].plot(d_acc_g_z, color='red')
        fig.tight_layout()
        fig.savefig('images/{}/pretraining'.format(folderpath))
        noise = lasagne.utils.floatX(np.random.normal(size=g_specs['noise_shape']))
        rand_ids = np.random.randint(0, g_specs['noise_shape'][0], 64)
        samples = g_sample_fn(d_X, noise, d_C, d_M)[rand_ids]

        plt.imsave('images/{}/pretraining_samples.png'.format(folderpath),
                   (samples.reshape(8, 8, max_len, n_features)
                           .transpose(0, 2, 1, 3)
                           .reshape(8*max_len, 8*n_features)).T,
                   cmap='gray',
                   origin='bottom')
        plt.close('all')


# training loop
d_losses = []
g_losses = []
d_acc_x = []
d_acc_g_z = []
print("Training {} epochs, {} samples each, batch size {}".format(
    n_epochs, epoch_size, d_batch_size))
for epoch in tqdm(range(n_epochs)):
    d_loss, acc_d_x, acc_d_g_z, g_loss = 0, 0, 0, 0
    for iteration in range(epoch_size):
        for i in range(n_d_iterations):
            # load mini-batch
            d_X, d_C, d_M = data_iter.next()
            g_C, g_M = d_C, d_M

            # randomly add noise to d_X
            if (np.random.random() > 0.66):
                g_X = np.copy(d_X)
                d_X += np.random.normal(0, 1, size=d_X.shape)
            else:
                g_X = d_X

        d_loss, acc_d_x, acc_d_g_z = d_train_fn(d_X, d_M, g_X, g_C, g_M)

        for i in range(n_g_iterations):
            # load mini-batch
            d_X, d_C, d_M = data_iter.next()
            g_X, g_C, g_M = d_X, d_C, d_M

            # train iteration
            if d_specs.get('unroll'):
                g_loss += g_train_fn(g_X, g_C, g_M, d_M, d_X)
            else:
                g_loss += g_train_fn(g_X, g_C, g_M)

    d_losses.append(d_loss)
    d_acc_x.append(acc_d_x)
    d_acc_g_z.append(acc_d_g_z)
    g_losses.append(g_loss)

    fig, axes = plt.subplots(3, 1, figsize=(8, 6))
    axes[0].set_title('Loss(d)')
    axes[0].plot(d_losses)
    axes[1].set_title('Loss(g)')
    axes[1].plot(g_losses)
    axes[2].set_title('Accuracy D(x) D(G(z)')
    axes[2].plot(d_acc_x, color='blue')
    axes[2].plot(d_acc_g_z, color='red')
    fig.tight_layout()
    fig.savefig('images/{}/epoch{}'.format(folderpath, epoch))

    noise = lasagne.utils.floatX(np.random.normal(size=g_specs['noise_shape']))
    rand_ids = np.random.randint(0, g_specs['noise_shape'][0], 64)
    samples = g_sample_fn(d_X, noise, d_C, d_M)[rand_ids]
    plt.imsave('images/{}/epoch{}_samples.png'.format(folderpath, epoch),
            (samples.reshape(8, 8, max_len, n_features)
                    .transpose(0, 2, 1, 3)
                    .reshape(8*max_len, 8*n_features)).T,
            cmap='gray',
            origin='bottom')

    plt.close('all')
    display.clear_output(wait=True)
