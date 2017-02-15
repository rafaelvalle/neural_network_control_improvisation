import os
from collections import defaultdict
import numpy as np
import glob2 as glob
import pdb


def encode_labels(labels, one_hot=False):
    labels_enc = []
    labels_dict = {}
    i = 0
    for l in labels:
        if l not in labels_dict:
            labels_dict[l] = i
            i += 1
        labels_enc.append(labels_dict[l])

    if one_hot:
        eye = np.eye(len(labels_dict), dtype=np.int32)
        labels_enc = eye[labels_enc]

    return labels_enc


def load_data(datapath, glob_file_str, n_pieces, crop=None, as_dict=True,
              scale=True, patch_size=False, encoded_labels=True):

    data = defaultdict(list)
    if not as_dict:
        data = []
        labels = []

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
            if not as_dict:
                if patch_size:
                    data.extend(cur_data)
                    labels.extend([composer] * len(cur_data))
                else:
                    data.append(cur_data)
                    labels.extend(composer)
            else:
                data[composer].append(cur_data)
    if not as_dict:
        return np.array(data), np.array(labels)

    return data
