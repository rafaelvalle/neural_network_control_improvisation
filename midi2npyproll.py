#!/usr/bin/python

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import argparse
import sys
import glob2 as glob
import numpy as np
import pretty_midi as pm
from music_utils import quantize, interpolate_between_beats

def main(globstr, beat_subdivisions, fs, quantized, wrap, save_img):
    for filepath in glob.glob(globstr):
        try:
            data = pm.PrettyMIDI(filepath)
            b = data.get_beats()
            beats = interpolate_between_beats(b, beat_subdivisions)
            if quantized:
                quantize(data, beats)
            if not fs:
                cur_fs = 1./beats[1]
                while cur_fs > wrap:
                    cur_fs = cur_fs * 0.5
            else:
                cur_fs = fs
            print("{}, {}".format(filepath, cur_fs))
            # proll = data.get_piano_roll(fs=fs, times=beats)
            proll = data.get_piano_roll(fs=cur_fs).astype(int)
            if np.isnan(proll).any():
                print("{} had NaN cells".format(filepath))
            # automatically appends .npy fo filename
            np.save(filepath, proll)
            # save image
            if save_img:
                plt.imsave(filepath+'.png', np.flipud(proll))
        except:
            print filepath, sys.exc_info()[0]
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("globstr", type=str,
                        help="Glob string")
    parser.add_argument(
        "-b", "--beat_subdivisions", type=int, default=12,
        help="Number of subdivisions per beat")
    parser.add_argument(
        "-f", "--fs", type=int, default=0,
        help="Sampling rate per second")
    parser.add_argument(
        "-q", "--quantized", type=int, default=0,
        help="Quantize to beat times and subdivisions")
    parser.add_argument(
        "-s", "--save_img", type=int, default=0,
        help="Save pianoroll image")
    parser.add_argument(
        "-w", "--wrap", type=int, default=20,
        help="Recursively wrap to half fs while larger than wrap value")

    args = parser.parse_args()
    print(args)
    main(args.globstr, args.beat_subdivisions, args.fs, args.quantized,
         args.wrap, args.save_img)
