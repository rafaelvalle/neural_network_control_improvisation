#!/usr/bin/python

import argparse
import numpy as np
from scipy.ndimage import imread
from music_utils import pianoroll_to_midi
from data_processing import postprocess_proll, offset_proll
import pdb

def main(filepath, shape, fs, program, threshold, boolean, argmax,
         offset):
    shape = np.array(shape.split(' '), dtype=np.float32)
    img = imread(filepath, flatten=True)
    n_rows = img.shape[0] / shape[0]
    n_cols = img.shape[1] / shape[1]

    i = 0
    for col_imgs in np.split(img, n_cols, axis=0):
        for proll in np.split(col_imgs, n_rows, axis=1):
            proll = proll / proll.max()
            proll = offset_proll(proll, offset)
            proll = postprocess_proll(proll, threshold, argmax, boolean)
            pianoroll_to_midi(
                proll, fs, program, filepath=filepath+'{}.midi'.format(i))
            i += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("filepath", type=str, help="Filepath")
    parser.add_argument("shape", type=str, help="Dimensions of patch, 64 64")
    parser.add_argument("fs", type=int, default=10,
                        help="Sampling rate per second")
    parser.add_argument("-p", "--program", type=int, default=1,
                        help="MIDI program")
    parser.add_argument("-t", "--threshold", type=lambda x: float(x), default=0,
                        help="Threshold for silence")
    parser.add_argument("-b", "--boolean", type=int, default=0,
                        help="Ignore velocity")
    parser.add_argument("-m", "--argmax", type=int, default=0,
                        help="Argmax per timestep")
    parser.add_argument("-o", "--offset", type=int, default=0,
                        help="Offset of piano roll lowest note")

    args = parser.parse_args()
    print(args)
    main(args.filepath, args.shape, args.fs, args.program, args.threshold,
         args.boolean, args.argmax, args.offset)
