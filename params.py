from __future__ import division
import lasagne

n_timesteps = 12*4
note_resolution = 128
n_pitches = n_timesteps * .5
offset = 60
viol_ratio = 0.5
n_obs = int(1e+4)
n_features = 1
max_len = n_timesteps
min_len = max_len // 2

# pitch sets
specs = {1: (((0, 4, 7, 10), 0), ),
         2: (((0, 4, 7, 10), 0), ((2, 5, 9, 11), n_timesteps//2)),
         3: (((0, 4, 7, 10), 0),
             ((0, 2, 5, 9), int(n_timesteps*.5))),
         4: (((0, 4, 7, 11), 0),
             ((2, 5, 7, 11), int(n_timesteps*.25)),
             ((0, 4, 7, 9), int(n_timesteps*.50)),
             ((0, 4, 5, 9), int(n_timesteps*.75)))}

# neural network parameter not to be explored with bayesian parameter estimation
nnet_params = {
    'conv_rnn': {'batch_size': 16,
                 'epoch_size': 128,
                 'input_shape': (16, n_timesteps, n_features),
                 'n_filters':  18,
                 'filter_size': (6, 1),
                 'mask_shape': (16, n_timesteps),
                 'n_hidden': 1024,
                 'grad_clip': 100,
                 'init': lasagne.init.HeUniform(),
                 'non_linearities': (lasagne.nonlinearities.rectify,  # conv
                                     lasagne.nonlinearities.tanh,  # feedforw
                                     lasagne.nonlinearities.tanh,  # feedbackd
                                     lasagne.nonlinearities.linear),  # output
                 'update_func': lasagne.updates.adadelta
                 },
    'rnn': {'batch_size': 16,
            'epoch_size': 128,
            'input_shape': (16, n_timesteps, n_features),
            'mask_shape': (16, n_timesteps),
            'n_hidden': 1024,
            'grad_clip': 100,
            'init': lasagne.init.HeUniform(),
            'non_linearities': (lasagne.nonlinearities.rectify,  # feedforward
                                lasagne.nonlinearities.rectify,  # feedbackward
                                lasagne.nonlinearities.linear),  # output layer
            'update_func': lasagne.updates.adadelta
            },
    'proll': {'n_layers': 4,
              'batch_size': 16,
              'epoch_size': 128,
              'widths': [None, 1024, 1024, n_timesteps*note_resolution],
              'non_linearities': (None,
                                  lasagne.nonlinearities.rectify,
                                  lasagne.nonlinearities.rectify,
                                  lasagne.nonlinearities.softmax),
              'update_func': lasagne.updates.adadelta
              },
    'seq': {'n_layers': 4,
            'batch_size': 16,
            'epoch_size': 128,
            'widths': [None, 1024, 1024, n_timesteps],
            'non_linearities': (None,
                                lasagne.nonlinearities.rectify,
                                lasagne.nonlinearities.rectify,
                                lasagne.nonlinearities.linear),
            'update_func': lasagne.updates.adadelta
            }
}

# random number seed
rand_num_seed = 1

# hyperparameter space to be explored using bayesian parameter optimization
hyperparameter_space = {
    'general_network': {
        'momentum': {'type': 'float', 'min': 0., 'max': 1.},
        'dropout': {'type': 'int', 'min': 0, 'max': 1},
        'learning_rate': {'type': 'float', 'min': .000001, 'max': .1},
        'network': {'type': 'enum', 'options': ['general_network']}
        },

    'rnn': {
        'momentum': {'type': 'float', 'min': 0., 'max': 1.},
        'learning_rate': {'type': 'float', 'min': .000001, 'max': .1},
        'network': {'type': 'enum', 'options': ['rnn_network']}
        },

    'conv_rnn': {
        'momentum': {'type': 'float', 'min': 0., 'max': 1.},
        'learning_rate': {'type': 'float', 'min': .000001, 'max': .1},
        'network': {'type': 'enum', 'options': ['conv_rnn_network']}
        }
    }
