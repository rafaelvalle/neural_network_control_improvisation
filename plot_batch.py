import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
from data_processing import load_proll_data, iterate_minibatches_proll

# load data
datapath = '/media/steampunkhd/rafaelvalle/datasets/MIDI/Piano'
glob_file_str = '*.npy'
n_pieces = 0  # 0 is equal to all pieces, unbalanced dataset
crop = (32, 96)
alphabet_size = 64
as_dict = False
n_steps = 64
i_len = 64
patch_size = False
inputs, labels = load_proll_data(
    datapath, glob_file_str, n_pieces, crop, as_dict, patch_size=patch_size)
labels = np.array(labels)
iterator = iterate_minibatches_proll
# shuffle data
np.random.shuffle(inputs)
BATCH_SIZE = 64
train_gen = iterator(inputs, labels, BATCH_SIZE, shuffle=True, length=i_len,
                     forever=True)
for i in range(10):
    samples, _ = train_gen.next()
    plt.imsave('real_sample_{}.png'.format(i),
               np.flipud((samples.reshape(8, 8, alphabet_size, n_steps)
                         .transpose(0, 2, 1, 3)
                         .reshape(8*alphabet_size, 8*n_steps))), cmap='bwr')
