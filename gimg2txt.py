#!/usr/bin/python

import argparse
import numpy as np
from scipy.ndimage import imread
import cPickle as pkl
import pdb


def main(filepath, shape, decoder_path):
    decoder = pkl.load(open(decoder_path, "rb"))
    shape = np.array(shape.split(' '), dtype=np.float32)
    img = imread(filepath, flatten=True)
    n_rows = int(img.shape[0] / shape[0])
    n_cols = int(img.shape[1] / shape[1])

    i = 0
    text = '{}\n'.format(filepath)
    for row_imgs in np.split(img, n_rows, axis=0):
        for col_img in np.split(row_imgs, n_cols, axis=1):
            cur_txt = decoder.decode(np.argmax(col_img, axis=0))
            cur_txt = ''.join(cur_txt)
            text += '{}, {}\n'.format(i, cur_txt)
            i += 1
    with open(filepath+'.txt', "w") as text_file:
        text_file.write(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("filepath", type=str, help="Filepath")
    parser.add_argument("shape", type=str, help="Dimensions of patch, 64 64")
    parser.add_argument("decoder", type=str, help="Decoder path")

    args = parser.parse_args()
    print(args)
    main(args.filepath, args.shape, args.decoder)
