import numpy as np
import theano


def get_next_batch(inputs, targets, batch_size, n_iters):
    for _ in range(n_iters):
        excerpt = np.random.permutation(len(inputs))[:batch_size]
        yield inputs[excerpt], targets[excerpt]


def get_next_batch_rnn(inputs, targets, masks, batch_size, n_iters, conds=None):
    for _ in range(n_iters):
        excerpt = np.random.permutation(len(inputs))[:batch_size]
        if conds is None:
            yield inputs[excerpt], targets[excerpt], masks[excerpt]
        else:
            yield inputs[excerpt], targets[excerpt], masks[excerpt], conds[excerpt]


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def zerosX(X):
    return np.zeros(X, dtype=theano.config.floatX)


def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))
