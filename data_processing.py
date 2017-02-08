import os
from collections import defaultdict
import numpy as np
import glob2 as glob


def load_data(datapath, glob_file_str, n_pieces, crop=None, as_dict=True,
              scale=True):
    data = []
    if as_dict:
        data = defaultdict(list)

    for folderpath in glob.glob(os.path.join(datapath, '*/')):
        composer = os.path.basename(os.path.normpath(folderpath))
        filepaths = glob.glob(os.path.join(
            os.path.join(datapath, composer), glob_file_str))
        if n_pieces:
            filepaths = np.random.choice(filepaths, n_pieces, replace=False)
        for filepath in filepaths:
            cur_data = np.load(filepath)
            if crop is not None:
                cur_data = cur_data[:, crop[0]:crop[1]]
            if scale:
                # scale each frame [-1, 1]
                cur_data += cur_data.min(axis=1)[:, None]
                cur_data /= cur_data.max(axis=1)[:, None]
                cur_data = np.nan_to_num(cur_data)
                cur_data = cur_data * 2 - 1
            if as_dict:
                data[composer].append(cur_data)
            else:
                data.append([cur_data, composer])
    return data
