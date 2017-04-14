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


def main(globstr, beat_subdivisions, save_img):
    for filepath in glob.glob(globstr):
        try:
            d = pm.PrettyMIDI(filepath)
            b = d.get_beats()
            beats = interpolate_between_beats(b, beat_subdivisions)
            quantize(d, beats)
            # fs = min(int(np.floor(1./beats[1]))+1, 20)
            fs = 1./beats[1]
            # p = d.get_piano_roll(fs=fs, times=beats)
            proll = d.get_piano_roll(fs=fs).astype(int)
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
        "-s", "--save_img", type=int, default=0,
        help="Save pianoroll image")

    args = parser.parse_args()
    print(args)
    main(args.globstr, args.beat_subdivisions, args.save_img)
