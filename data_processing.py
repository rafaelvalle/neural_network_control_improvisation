import os
from collections import defaultdict
import numpy as np
import glob2 as glob
import pdb


def load_data(datapath, glob_file_str, n_pieces, crop=None, as_dict=True,
              scale=True, patch_size=False, with_labels=False):
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
            if patch_size:
                ids = np.arange(0, len(cur_data) - patch_size, patch_size)
                cur_data = np.array([
                    cur_data[ids[i-1]:ids[i]] for i in range(1, len(ids))])
            if as_dict:
                data[composer].append(cur_data)
            elif patch_size and not with_labels:
                data.extend(cur_data)
            elif with_labels:
                data.extend([cur_data, composer])
            else:
                data.append(cur_data)
    if not as_dict:
        data = np.array(data)
    return data
