""" Adaptep from Colin Raffel's git repo https://github.com/craffel/ """
import numpy as np
import theano
from theano import tensor as T
import lasagne
import nnet_utils
# import seaborn
import matplotlib
matplotlib.use('Agg')
import pylab as plt

import pdb


def train_sequence_rnn(data, layers, updates_fn, batch_size=16, epoch_size=128,
                       initial_patience=1000, improvement_threshold=0.99,
                       patience_increase=5, max_iter=100000):

    # get input and mask vars from layers and specifiy output var
    input_var = layers[0].input_var
    mask_var = layers[1].input_var
    target_var = T.vector('target')

    # create a cost expression for training
    prediction = lasagne.layers.get_output(layers[2])
    cost = T.mean((prediction.flatten() - target_var)**2)
    # create parameter update expressions for training
    params = lasagne.layers.get_all_params(layers[2], trainable=True)
    print("Computing updates ...")
    updates = updates_fn(cost, params)

    # compile functions for performing training step and returning
    # corresponding training cost
    print("Compiling functions ...")
    train_fn = theano.function(inputs=[input_var, target_var, mask_var],
                               outputs=cost,
                               updates=updates)

    # create cost expression for validation
    # deterministic forward pass to disable droupout layers
    val_prediction = lasagne.layers.get_output(layers[2], deterministic=True)
    val_cost = T.mean((val_prediction.flatten() - target_var)**2)
    # val_output = theano.function([input_var, mask_var], val_prediction)
    # compile a function to compute the validation cost and objective function
    validate_fn = theano.function(inputs=[input_var, target_var, mask_var],
                                  outputs=val_cost)

    # create data iterators
    print("Create data iterators")
    train_data_iter = nnet_utils.get_next_batch_rnn(
        data['train']['without_specs'], data['train']['with_specs'],
        data['train']['masks'], batch_size, max_iter)
    patience = initial_patience
    current_val_cost = np.inf
    train_cost = 0.0
    print("Training and Validating ...")
    for n, (x_batch, y_batch, mask_batch) in enumerate(train_data_iter):
        train_cost += train_fn(x_batch, y_batch, mask_batch)

        # Stop training if NaN is encountered
        if not np.isfinite(train_cost):
            print 'Bad training er {} at iteration {}'.format(train_cost, n)
            break

        if n and not (n % epoch_size):
            epoch_result = {'iteration': n,
                            'train_cost': train_cost / float(epoch_size),
                            'validate_cost': 0.0,
                            'validate_objective': 0.0}
            # compute validation cost and objective
            cost = np.float(validate_fn(data['validate']['without_specs'],
                                        data['validate']['with_specs'],
                                        data['validate']['masks']))
            epoch_result['validate_cost'] = cost
            epoch_result['validate_objective'] = cost

            """
            n_obs = 5000
            ids = np.random.randint(
                0, len(data['validate']['without_specs']), n_obs)
            outs = val_output(data['validate']['without_specs'][ids],
                              data['validate']['masks'][ids])

            plt.subplot(3,1,1)
            plt.plot(data['validate']['with_specs'][ids], 'g.')
            plt.subplot(3,1,2)
            plt.plot(outs, 'r.')
            plt.subplot(3,1,3)
            plt.plot(data['validate']['with_specs'][ids] - outs, 'b.')
            plt.savefig('img.png')
            plt.close()
            """

            # Test whether this validate cost is the new smallest
            if epoch_result['validate_cost'] < current_val_cost:
                # To update patience, we must be smaller than
                # improvement_threshold*(previous lowest validation cost)
                patience_cost = improvement_threshold*current_val_cost
                if epoch_result['validate_cost'] < patience_cost:
                    # Increase patience by the supplied about
                    patience += epoch_size*patience_increase
                # Even if we didn't increase patience, update lowest valid cost
                current_val_cost = epoch_result['validate_cost']
            # Store patience after this epoch
            epoch_result['patience'] = patience

            if n > patience:
                break

            yield epoch_result


def train_rnn_proll(data_iter, layers, updates_fn, batch_size=32,
                    epoch_size=128, initial_patience=1000,
                    improvement_threshold=0.99, patience_increase=5,
                    max_iter=100000):
    # get input and mask vars from layers and specifiy output var
    input_var = layers[0].input_var
    mask_var = layers[1].input_var
    target_var = T.imatrix('target')

    # create a cost expression for training
    prediction = lasagne.layers.get_output(layers[2])
    cost = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    cost = cost.mean()

    # create parameter update expressions for training, compile func for train
    params = lasagne.layers.get_all_params(layers[2], trainable=True)
    updates = updates_fn(cost, params)
    train_fn = theano.function(inputs=[input_var, target_var, mask_var],
                               outputs=cost,
                               updates=updates)
    # train_pred_fn = theano.function([input_var, mask_var], prediction)

    # create cost expression for validation
    # deterministic forward pass to disable droupout layers
    val_prediction = lasagne.layers.get_output(layers[2], deterministic=True)
    val_cost = lasagne.objectives.categorical_crossentropy(
        val_prediction, target_var)
    val_cost = val_cost.mean(keepdims=True)
    val_err_rate = T.mean(
        T.neq(T.argmax(val_prediction, axis=1), T.argmax(target_var, axis=1)),
        keepdims=True)

    # compile funcs for validation cost and prediction
    validate_fn = theano.function(
        inputs=[input_var, target_var, mask_var],
        outputs=[val_cost, val_err_rate])
    # val_pred_fn = theano.function([input_var, mask_var], val_prediction)

    # create data iterators
    patience = initial_patience
    current_val_cost = np.inf
    train_cost = 0.0
    print("Training and Validating ...")
    for n, (x_batch, y_batch, mask_batch) in enumerate(data_iter['train']):
        train_cost_cur = train_fn(x_batch, y_batch, mask_batch)
        train_cost += train_cost_cur
        # train_pred = train_pred_fn(x_batch, mask_batch)

        # Stop training if NaN is encountered
        if not np.isfinite(train_cost):
            print 'Bad training er {} at iteration {}'.format(train_cost, n)
            break

        if n and not (n % epoch_size):
            # compute validation cost and objective
            val_input, val_tgt, val_mask = data_iter['valid'].next()
            val_cost, val_err_rate = validate_fn(val_input, val_tgt, val_mask)
            val_cost = val_cost[0]
            val_err_rate = val_err_rate[0]

            epoch_result = {'iteration': n,
                            'train_cost_cur': train_cost_cur,
                            'train_cost': train_cost / float(n),
                            'validate_cost': val_cost,
                            'validate_objective': val_err_rate}

            # compute predictions
            # predictions = val_pred_fn(val_input, val_mask)
            # pdb.set_trace()

            # Test whether this validate cost is the new smallest
            if epoch_result['validate_cost'] < current_val_cost:
                # To update patience, we must be smaller than
                # improvement_threshold * (previous lowest validation cost)
                patience_cost = improvement_threshold * current_val_cost
                if epoch_result['validate_cost'] < patience_cost:
                    # Increase patience by the supplied about
                    patience += epoch_size * patience_increase
                # Even if we didn't increase patience, update lowest valid cost
                current_val_cost = epoch_result['validate_cost']
            # Store patience after this epoch
            epoch_result['patience'] = patience

            if n > patience:
                break

            yield epoch_result


def train_sequence(data, layers, updates_fn, batch_size=16, epoch_size=128,
                   initial_patience=1000, improvement_threshold=0.99,
                   patience_increase=5, max_iter=100000):

    # specify input and target theano data types
    input_var = T.matrix('inputs')
    target_var = T.matrix('targets')

    # create a cost expression for training
    prediction = lasagne.layers.get_output(layers, input_var)
    cost = lasagne.objectives.squared_error(prediction, target_var)
    cost = cost.mean()

    # create parameter update expressions for training
    params = lasagne.layers.get_all_params(layers, trainable=True)
    updates = updates_fn(cost, params)

    # compile functions for performing training step and returning
    # corresponding training cost
    train_fn = theano.function(inputs=[input_var, target_var],
                               outputs=cost,
                               updates=updates)

    # create cost expression for validation
    # deterministic forward pass to disable droupout layers
    val_prediction = lasagne.layers.get_output(layers, input_var,
                                               deterministic=True)
    val_cost = lasagne.objectives.squared_error(val_prediction, target_var)
    val_cost = val_cost.mean()

    # val_output = theano.function([input_var], val_prediction)
    # compile a function to compute the validation cost and objective function
    validate_fn = theano.function(inputs=[input_var, target_var],
                                  outputs=val_cost)

    # create data iterators
    train_data_iter = nnet_utils.get_next_batch(data['train']['without_specs'],
                                                data['train']['with_specs'],
                                                batch_size, max_iter)

    patience = initial_patience
    current_val_cost = np.inf
    train_cost = 0.0

    for n, (x_batch, y_batch) in enumerate(train_data_iter):
        train_cost += train_fn(x_batch, y_batch)

        # Stop training if NaN is encountered
        if not np.isfinite(train_cost):
            print 'Bad training er {} at iteration {}'.format(train_cost, n)
            break

        if n and not (n % epoch_size):
            epoch_result = {'iteration': n,
                            'train_cost': train_cost / float(epoch_size),
                            'validate_cost': 0.0,
                            'validate_objective': 0.0}

            # compute validation cost and objective
            cost = np.float(validate_fn(data['validate']['without_specs'],
                                        data['validate']['with_specs']))

            epoch_result['validate_cost'] = cost
            epoch_result['validate_objective'] = cost

            # Test whether this validate cost is the new smallest
            if epoch_result['validate_cost'] < current_val_cost:
                # To update patience, we must be smaller than
                # improvement_threshold*(previous lowest validation cost)
                patience_cost = improvement_threshold*current_val_cost
                if epoch_result['validate_cost'] < patience_cost:
                    # Increase patience by the supplied about
                    patience += epoch_size*patience_increase
                # Even if we didn't increase patience, update lowest valid cost
                current_val_cost = epoch_result['validate_cost']
            # Store patience after this epoch
            epoch_result['patience'] = patience

            if n > patience:
                break

            yield epoch_result


def train_proll(data, layers, updates_fn, batch_size=16, epoch_size=128,
                initial_patience=1000, improvement_threshold=0.99,
                patience_increase=5, max_iter=100000):

    # specify input and target theano data types
    input_var = T.matrix('inputs')
    target_var = T.matrix('targets')

    # create a cost expression for training
    prediction = lasagne.layers.get_output(layers, input_var)
    cost = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    cost = cost.mean()

    # create parameter update expressions for training
    params = lasagne.layers.get_all_params(layers, trainable=True)
    updates = updates_fn(cost, params)

    # compile functions for performing training step and returning
    # corresponding training cost
    train_fn = theano.function(inputs=[input_var, target_var],
                               outputs=cost,
                               updates=updates)

    # create cost expression for validation
    # deterministic forward pass to disable droupout layers
    val_prediction = lasagne.layers.get_output(layers, input_var,
                                               deterministic=True)
    val_cost = lasagne.objectives.categorical_crossentropy(
        val_prediction, target_var)
    val_cost = val_cost.mean()
    # val_acc = T.mean(T.eq(T.argmax(val_prediction, axis=1), target_var))

    # compile a function to compute the validation cost and objective function
    validate_fn = theano.function(inputs=[input_var, target_var],
                                  outputs=val_cost)

    # build a prediction function to manually evaluate output
    val_output = theano.function([input_var], val_prediction)

    # create data iterators
    train_data_iter = nnet_utils.get_next_batch(data['train']['without_specs'],
                                                data['train']['with_specs'],
                                                batch_size, max_iter)

    patience = initial_patience
    current_val_cost = np.inf
    train_cost = 0.0

    for n, (x_batch, y_batch) in enumerate(train_data_iter):
        train_cost += train_fn(x_batch, y_batch)

        # Stop training if NaN is encountered
        if not np.isfinite(train_cost):
            print 'Bad training er {} at iteration {}'.format(train_cost, n)
            break

        if n and not (n % epoch_size):
            epoch_result = {'iteration': n,
                            'train_cost': train_cost / float(epoch_size),
                            'validate_cost': 0.0,
                            'validate_objective': 0.0}

            # compute validation cost and objective
            cost = np.float(validate_fn(data['validate']['without_specs'],
                                        data['validate']['with_specs']))

            epoch_result['validate_cost'] = cost
            epoch_result['validate_objective'] = cost

            if cost < 13:
                pred = val_output(data['validate']['without_specs'])[0]
                real = data['validate']['with_specs'][0]

                plt.figure()
                plt.subplot(121)
                plt.imshow(pred.reshape(48, 128).T, aspect='auto')
                plt.subplot(122)
                plt.imshow(real.reshape(48, 128).T, aspect='auto')
                plt.show()
            # Test whether this validate cost is the new smallest
            if epoch_result['validate_cost'] < current_val_cost:
                # To update patience, we must be smaller than
                # improvement_threshold*(previous lowest validation cost)
                patience_cost = improvement_threshold*current_val_cost
                if epoch_result['validate_cost'] < patience_cost:
                    # Increase patience by the supplied about
                    patience += epoch_size*patience_increase
                # Even if we didn't increase patience, update lowest valid cost
                current_val_cost = epoch_result['validate_cost']
            # Store patience after this epoch
            epoch_result['patience'] = patience

            if n > patience:
                break

            yield epoch_result


def build_conv_rnn(input_shape, n_filters, filter_size, mask_shape, n_hidden,
                   grad_clip, init, non_linearities):
    # input layer
    l_in = lasagne.layers.InputLayer(shape=input_shape)
    # convolutional layer
    l_conv = lasagne.layers.Conv2DLayer(
        l_in, num_filters=n_filters, filter_size=filter_size,
        nonlinearity=non_linearities[0],
        W=lasagne.init.HeNormal(gain='relu'))
    # mask determines which indices are part of the sequence for each batch
    l_mask = lasagne.layers.InputLayer(shape=mask_shape)
    # bidirectional network
    l_forward = lasagne.layers.RecurrentLayer(
        l_conv, n_hidden, mask_input=l_mask, grad_clipping=grad_clip,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=non_linearities[1], only_return_final=True)
    l_backward = lasagne.layers.RecurrentLayer(
        l_conv, n_hidden, mask_input=l_mask, grad_clipping=grad_clip,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=non_linearities[2], only_return_final=True,
        backwards=True)
    # concatenate output of forward and backward layers
    l_concat = lasagne.layers.ConcatLayer([l_forward, l_backward])
    # output is one dense layer with one output unit
    l_out = lasagne.layers.DenseLayer(
        l_concat, num_units=1, nonlinearity=non_linearities[3])
    return l_in, l_mask, l_out


def build_rnn(input_shape, mask_shape, n_hidden, grad_clip, init,
              non_linearities):
    # input layer
    l_in = lasagne.layers.InputLayer(shape=input_shape)
    # mask determines which indices are part of the sequence for each batch

    l_mask = lasagne.layers.InputLayer(shape=mask_shape,
                                       input_var=T.imatrix())
    # bidirectional network
    l_forward = lasagne.layers.RecurrentLayer(
        l_in, n_hidden, mask_input=l_mask, grad_clipping=grad_clip,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=non_linearities[0], only_return_final=True)
    l_backward = lasagne.layers.RecurrentLayer(
        l_in, n_hidden, mask_input=l_mask, grad_clipping=grad_clip,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=non_linearities[1], only_return_final=True,
        backwards=True)
    # concatenate output of forward and backward layers
    l_concat = lasagne.layers.ConcatLayer([l_forward, l_backward])
    # output is one dense layer with one output unit
    l_out = lasagne.layers.DenseLayer(
        l_concat, num_units=1, nonlinearity=non_linearities[2])
    return l_in, l_mask, l_out


def build_rnn_proll(input_shape, mask_shape, n_hidden, grad_clip, init,
                    non_linearities):
    # input layer
    l_in = lasagne.layers.InputLayer(shape=input_shape)
    # mask determines which indices are part of the sequence for each batch
    l_mask = lasagne.layers.InputLayer(shape=mask_shape)
    # bidirectional network
    l_forward = lasagne.layers.RecurrentLayer(
        l_in, n_hidden, mask_input=l_mask, grad_clipping=grad_clip,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=non_linearities[0], only_return_final=True)
    l_backward = lasagne.layers.RecurrentLayer(
        l_in, n_hidden, mask_input=l_mask, grad_clipping=grad_clip,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=non_linearities[1], only_return_final=True,
        backwards=True)
    # concatenate output of forward and backward layers
    l_concat = lasagne.layers.ConcatLayer([l_forward, l_backward])
    # output is one dense layer with n ouput units
    l_out = lasagne.layers.DenseLayer(
        l_concat, num_units=12, nonlinearity=non_linearities[2])
    return l_in, l_mask, l_out


def build_general_network(input_shape, n_layers, widths,
                          non_linearities, drop_out=True):
    """
    Parameters
    ----------
    input_shape : tuple of int or None (batchsize, rows, cols)
        Shape of the input. Any element can be set to None to indicate that
        dimension is not fixed at compile time
    """

    # GlorotUniform is the default mechanism for initializing weights
    for i in range(n_layers):
        if i == 0:  # input layer
            layers = lasagne.layers.InputLayer(shape=input_shape)
        else:  # hidden and output layers
            layers = lasagne.layers.DenseLayer(layers,
                                               num_units=widths[i],
                                               nonlinearity=non_linearities[i])
            if drop_out and i < n_layers-1:  # output layer has no dropout
                layers = lasagne.layers.DropoutLayer(layers, p=0.5)

    return layers
