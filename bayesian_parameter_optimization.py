""" Adaptep from Colin Raffel's git repo https://github.com/craffel/
 Shared utility functions for downsampled hash sequence experiments.
 """

import os
import sys
import glob
import pdb
import numpy as np
import lasagne
import deepdish as dd
import traceback
import functools
import simple_spearmint
import neural_networks


def run_trial(data, nnet_params, hyperparameter_space, train_function):
    """Train a network given the task and hyperparameters and return the result.

    Parameters
    ----------
    data: np.ndarray
        dataset from which train and validate set will be built using k-fold
        cross validation. Last column must be target variable.
    nnet_params: dict
        Hyperparameter values that are not going to be optimized but parametrize
        the neural network.
    hyperparameter_space : dict
        Dictionary of model hyperparameters
    train_function : callable
        This function will be called with the constructed network, training
        data, and hyperparameters to create a model.

    Returns
    -------
    best_objective : float
        Lowest objective value achieved.
    best_epoch : dict
        Statistics about the epoch during which the lowest objective value was
        achieved.
    best_params : dict
        Parameters of the model for the best-objective epoch.
    """
    # We will be modifying params, so make a copy of it
    hyperparameter_space = dict(hyperparameter_space)
    print ',\n'.join(['\t{} : {}'.format(k, v)
                      for k, v in hyperparameter_space.items()])

    # Choose network structure based on network param
    if hyperparameter_space['network'] == 'general_network':
        build_network_layers = neural_networks.build_general_network
    elif hyperparameter_space['network'] == 'rnn_network':
        build_network_layers = neural_networks.build_rnn
    elif hyperparameter_space['network'] == 'rnn_proll_network':
        build_network_layers = neural_networks.build_rnn_proll
    elif hyperparameter_space['network'] == 'conv_rnn_network':
        build_network_layers = neural_networks.build_conv_rnn
    else:
        raise ValueError('Unknown network {}'.format(
            hyperparameter_space['network']))

    # Create train and validation datasets
    if hyperparameter_space['network'] in ('rnn_network', 'conv_rnn_network'):
        train_ids = np.random.binomial(1, .8, len(data['with_specs']))
        train_ids = train_ids.astype(bool)
        data = {'train': {'without_specs': data['without_specs'][train_ids],
                          'with_specs': data['with_specs'][train_ids],
                          'masks': data['masks'][train_ids]
                          },
                'validate': {'without_specs': data['without_specs'][~train_ids],
                             'with_specs': data['with_specs'][~train_ids],
                             'masks': data['masks'][~train_ids]
                             }
                }
    elif hyperparameter_space['network'] in ('rnn_proll_network', ):
        data = data
    else:
        train_ids = np.random.binomial(1, .8, len(data['with_specs']))
        train_ids = train_ids.astype(bool)
        data = {'train': {'without_specs': data['without_specs'][train_ids],
                          'with_specs': data['with_specs'][train_ids]
                          },
                'validate': {'without_specs': data['without_specs'][~train_ids],
                             'with_specs': data['with_specs'][~train_ids]
                             }
                }

    # build the neural network models
    if hyperparameter_space['network'] in ('rnn_network', 'rnn_proll_network'):
        # in rnn layers include l_in, l_mask and l_out
        layers = build_network_layers(
            nnet_params['input_shape'],
            nnet_params['mask_shape'],
            nnet_params['n_hidden'],
            nnet_params['grad_clip'],
            nnet_params['init'],
            nnet_params['non_linearities'])
    elif hyperparameter_space['network'] == 'conv_rnn_network':
        # in conv rnn layers include l_in, l_mask and l_out
        layers = build_network_layers(
            nnet_params['input_shape'],
            nnet_params['n_filters'],
            nnet_params['filter_size'],
            nnet_params['mask_shape'],
            nnet_params['n_hidden'],
            nnet_params['grad_clip'],
            nnet_params['init'],
            nnet_params['non_linearities'])
    else:
        layers = build_network_layers(
            (nnet_params['batch_size'],
             data['train']['without_specs'].shape[1]),
            nnet_params['n_layers'],
            nnet_params['widths'],
            nnet_params['non_linearities'],
            drop_out=hyperparameter_space['dropout'])

    # Generate updates-creating function
    updates_function = functools.partial(
        nnet_params['update_func'],
        learning_rate=hyperparameter_space['learning_rate'],
        rho=hyperparameter_space['momentum'])

    # Create a list of epochs and keep track of lowest objective found so far
    epochs = []
    best_objective = np.inf
    try:
        for epoch in train_function(data, layers, updates_function,
                                    nnet_params['batch_size'],
                                    nnet_params['epoch_size']):
            # Stop training if a nan training cost is encountered
            if not np.isfinite(epoch['train_cost']):
                break
            epochs.append(epoch)
            if epoch['validate_objective'] < best_objective:
                best_objective = epoch['validate_objective']
                best_epoch = epoch
                best_model = lasagne.layers.get_all_param_values(layers)
            print "{}: {}, {}, {} ".format(
                epoch['iteration'],
                epoch['train_cost'],
                epoch['validate_cost'],
                epoch['validate_objective'])
            sys.stdout.flush()
    # If there was an error while training, report it to whetlab
    except Exception:
        print "ERROR: "
        print traceback.format_exc()
        return np.nan, {}, {}
    print
    # Check that all training costs were not NaN; return NaN if any were.
    success = np.all([np.isfinite(e['train_cost']) for e in epochs])
    if np.isinf(best_objective) or len(epochs) == 0 or not success:
        print '    Failed to converge.'
        print
        return np.nan, {}, {}
    else:
        for k, v in best_epoch.items():
            print "\t{:>35} | {}".format(k, v)
        print
        return best_objective, best_epoch, best_model


def parameter_search(data, nnet_params, hyperparameter_space, trial_directory,
                     model_directory, train_function):
    """Run parameter optimization given some train function, writing out results
    Parameters
    ----------
    data: np.ndarray
        Matrix where rows are observations and columns are feature values.
        Last column must be target value.
        The data will be use to create a randomized train and validate set.
    nnet_params: dict
        Hyperparameter values that are not going to be optimized but parametrize
        the neural network.
    hyperparameter_space : dict
        Hyperparameter space (in the format used by `simple_spearmint`) to
        optimize over.
    trial_directory : str
        Directory where parameter optimization trial results will be written.
    model_directory : str
        Directory where the best-performing model will be written
    train_function : callable
        This function will be called with the constructed network, training
        data, and hyperparameters to create a model.
    """
    # Create parameter trials directory if it doesn't exist
    if not os.path.exists(trial_directory):
        os.makedirs(trial_directory)
    # Create model directory if it doesn't exist
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    # Create SimpleSpearmint suggester instance
    ss = simple_spearmint.SimpleSpearmint(hyperparameter_space)
    # Load in previous results for "warm start"
    for trial_file in glob.glob(os.path.join(trial_directory, '*.h5')):
        trial = dd.io.load(trial_file)
        ss.update(trial['hyperparameters'], trial['best_objective'])
    # Run parameter optimization forever
    while True:
        # Get a new suggestion
        suggestion = ss.suggest()
        # Train a network with these hyperparameters
        best_objective, best_epoch, best_model = run_trial(
            data, nnet_params, suggestion, train_function)
        # Update spearmint on the result
        ss.update(suggestion, best_objective)
        # Write out a result file
        trial_filename = ','.join('{}={}'.format(k, v)
                                  for k, v in suggestion.items()) + '.h5'
        dd.io.save(os.path.join(trial_directory, trial_filename),
                   {'hyperparameters': suggestion,
                    'best_objective': best_objective,
                    'best_epoch': best_epoch})
        # Also write out the entire model when the objective is the smallest
        # We don't want to write all models; they are > 100MB each
        if (not np.isnan(best_objective) and
                best_objective == np.nanmin(ss.objective_values)):
            dd.io.save(
                os.path.join(model_directory, 'best_model.h5'), best_model)
