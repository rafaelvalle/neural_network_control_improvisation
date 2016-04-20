""" Search for good hyperparameters for classifiction using the manually
perturbed (missing data) ADULT dataset.
"""
from __future__ import division
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import neural_networks
import bayesian_parameter_optimization as bpo
from params import nnet_params, hyperparameter_space
from params import specs, n_pitches, n_timesteps, offset, n_obs
from params import min_len, max_len
from music_utils import generateData, generateDataRNN


def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

RESULTS_PATH = 'results/'

if __name__ == '__main__':
    # Construct paths
    trial_directory = os.path.join(RESULTS_PATH, 'parameter_trials')
    model_directory = os.path.join(RESULTS_PATH, 'model')

    # Generate training dataset
    model_name = 'rnn'
    experiment = 1
    spec = specs[2]
    as_proll = False

    if model_name == 'proll':
        data = generateData(experiment, spec, n_pitches, n_timesteps, offset,
                            n_obs, as_proll)
        # reshape and normalize data
        for k, v in data.items():
            data[k] = data[k].reshape(
                data[k].shape[0], data[k].shape[1]*data[k].shape[2])
            # add epsilon to avoid zero probabilities
            data[k] = data[k].astype('float32') + np.finfo(float).eps
            data[k] /= np.sum(data[k][0])

        # Run parameter optimization forever
        nnet_params = nnet_params['proll']
        hyperparameter_space = hyperparameter_space['general_network']
        train_fn = neural_networks.train_proll
    elif model_name == 'conv_rnn':
        data = generateDataRNN(experiment, spec, n_pitches, n_timesteps,
                               offset, n_obs, min_len, max_len, as_proll)
        nnet_params = nnet_params['conv_rnn']
        hyperparameter_space = hyperparameter_space['conv_rnn']
        train_fn = neural_networks.train_sequence_rnn
    elif model_name == 'rnn':
        data = generateDataRNN(experiment, spec, n_pitches, n_timesteps,
                               offset, n_obs, min_len, max_len, as_proll)
        nnet_params = nnet_params['rnn']
        hyperparameter_space = hyperparameter_space['rnn']
        train_fn = neural_networks.train_sequence_rnn
    elif model_name == 'seq':
        data = generateDataRNN(experiment, spec, n_pitches, n_timesteps,
                               offset, n_obs, as_proll)
        nnet_params = nnet_params['seq']
        hyperparameter_space = hyperparameter_space['general_network']
        train_fn = neural_networks.train_sequence

    # Run parameter optimization forever
    bpo.parameter_search(data,
                         nnet_params,
                         hyperparameter_space,
                         os.path.join(trial_directory, model_name),
                         model_directory,
                         train_fn,
                         model_name)
