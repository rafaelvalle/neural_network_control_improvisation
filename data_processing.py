import os
from datetime import datetime as dt
from collections import defaultdict
import numpy as np
import pandas as pd
import glob2 as glob
import pdb

def iterate_minibatches_proll(inputs, labels, batch_size, shuffle=True,
                              forever=True, length=128):
    if shuffle:
        indices = np.arange(len(inputs))
    while True:
        if shuffle:
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            if length > 0:
                data = []
                # select random slice from each piano roll
                for i in excerpt:
                    rand_start = np.random.randint(
                        0, inputs[i].shape[1] - length)
                    data.append(inputs[i][:, rand_start:rand_start+length])
            else:
                data = inputs[excerpt]
            yield np.array(data).astype(np.float32), labels[excerpt]

        if not forever:
            break


def iterate_minibatches_text(inputs, labels, batch_size, encoder, shuffle=True,
                             forever=True, length=128, alphabet_size=128,
                             padding=None):
    from text_utils import binarizeText
    if shuffle:
        indices = np.arange(len(inputs))
    while True:
        if shuffle:
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)

            data = []
            # select random slice from each piano roll
            for i in excerpt:
                cur_data = binarizeText(
                    inputs[i], encoder, lower_case=True, remove_html=False)
                if cur_data.shape[1] < length:
                    if padding is None:
                        # ignore examples shorter than length
                        continue
                    elif padding == 'zero':
                        # zero pad data if shorter than length
                        z = np.zeros((alphabet_size, length))
                        z[:, :cur_data.shape[1]] = cur_data
                    elif padding == 'noise':
                        z = np.zeros((alphabet_size, length))
                        z[:, :cur_data.shape[1]] = cur_data
                        z[:, cur_data.shape[1]:] = np.random.normal(
                            0, 0.0001,
                            (cur_data.shape[0], length - cur_data.shape[1]))
                    elif padding == 'repeat':
                        z = np.zeros((alphabet_size, length))
                        z[:, :cur_data.shape[1]] = cur_data
                        # change to vector operation
                        for k in range(1, length - cur_data.shape[1]-1):
                            z[:, cur_data.shape[1] + k] = cur_data[
                                :, (k-1) % cur_data.shape[1]]
                    else:
                        raise Exception(
                            "Padding {} not supported".format(padding))
                    cur_data = z
                    data.append(cur_data)
                elif cur_data.shape[1] > length:
                    # slice data if larger than length
                    rand_start = np.random.randint(
                        0, cur_data.shape[1] - length)
                    cur_data = cur_data[:, rand_start:rand_start+length]
                    data.append(cur_data)
                else:
                    data.append(cur_data)
            # scale to [-1, 1]
            data = np.array(data)
            data += data.min()
            data /= data.max()
            data = (data * 2) - 1
            yield data.astype(np.float32), labels[excerpt]

        if not forever:
            break


def create_folder_structure(data_type, loss_type):
    if not os.path.exists(data_type):
        os.makedirs(data_type)
    if not os.path.exists(data_type+"/"+loss_type):
        os.makedirs(data_type+"/"+loss_type)

    folder_name = "{}/{}/{}".format(
        data_type, loss_type,
        dt.now().strftime('%Y_%m_%d_%H_%M_%S'))
    os.makedirs(folder_name)
    os.makedirs(folder_name + "/models")
    os.makedirs(folder_name + "/images")
    os.makedirs(folder_name + "/samples")
    return folder_name


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


def load_text_data(datapaths, data_col, label_col, n_pieces, as_dict=True,
                   patch_size=False, sep=','):

    data = defaultdict(list)
    if not as_dict:
        data = []
        labels = []

    for i in range(len(datapaths)):
        dataset = pd.read_csv(datapaths[i], sep=sep).as_matrix()
        if n_pieces:
            dataset = dataset[
                np.random.choice(np.arange(len(data)), n_pieces, replace=False)]
        for j in range(len(dataset)):
            cur_data = dataset[j, data_col[i]]
            cur_lbl = dataset[j, label_col[i]]
            if patch_size:
                if len(cur_data) < 2*patch_size:
                    cur_data = [cur_data[:patch_size]]
                else:
                    ids = np.arange(0, len(cur_data) - patch_size, patch_size)
                    cur_data = [
                        cur_data[ids[k-1]:ids[k]] for k in range(1, len(ids))]
            if not as_dict:
                if patch_size:
                    for l in range(len(cur_data)):
                        data.append(cur_data[l])
                    labels.extend([cur_lbl] * len(cur_data))
                else:
                    data.append(cur_data)
                    labels.extend(cur_lbl)
            else:
                data[cur_lbl].append(cur_data)
    if not as_dict:
        return np.array(data), np.array(labels)

    return data


def load_proll_data(datapath, glob_file_str, n_pieces, crop=None, as_dict=True,
                    scale=True, patch_size=False):

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
                cur_data = cur_data[crop[0]:crop[1], :]
            if scale:
                # scale each frame [-1, 1]
                #cur_data += cur_data.min(axis=1)[:, None]
                #cur_data /= cur_data.max(axis=1)[:, None]
                # global scaling
                cur_data += cur_data.min()
                cur_data = cur_data / float(cur_data.max())
                cur_data = np.nan_to_num(cur_data)
                cur_data = cur_data * 2 - 1
            if patch_size:
                ids = np.arange(0, cur_data.shape[1] - patch_size, patch_size)
                cur_data = np.array([
                    cur_data[:, ids[i-1]:ids[i]] for i in range(1, len(ids))])
            if not as_dict:
                if patch_size:
                    data.extend(cur_data)
                    labels.extend([composer] * len(cur_data))
                else:
                    data.append(cur_data)
                    labels.append(composer)
            else:
                data[composer].append(cur_data)
    if not as_dict:
        if patch_size:
            return np.array(data), np.array(labels)
        else:
            return data, np.array(labels)

    return data


def postprocess_proll(proll, threshold, argmax, boolean):
    if threshold < proll.max():
        proll[proll < threshold] = -1
    if argmax:
        z = np.zeros(proll.shape) - 1
        max_per_col = np.argmax(proll, axis=0)
        z[max_per_col, np.arange(proll.shape[1])] = (
            proll[max_per_col, np.arange(proll.shape[1])])
        proll = z
    proll += abs(proll.min())
    if proll.max() != 0:
        proll /= proll.max()
    if boolean:
        proll = (proll > 0).astype(int)
    proll *= 127
    proll = proll.astype(int)
    return proll


def offset_proll(proll, offset):
    score = np.zeros((128, proll.shape[1])) - 1
    score[offset:offset+proll.shape[0]] = proll
    return score
