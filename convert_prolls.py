#!/usr/bin/python

import argparse
import numpy as np
from music_utils import pianoroll_to_midi
import glob2 as glob


def convert(globstr, fs, program, threshold, samples, boolean):
    for filepath in glob.glob(globstr):
        prolls = np.load(filepath)
        print('{}, {}'.format(filepath, prolls.shape))
        for i in range(min(samples, len(prolls))):
            proll = prolls[i]
            if len(proll.shape) == 3:
                proll = proll[0]
            if threshold is not None:
                proll[proll < threshold] = -1
            proll += abs(proll.min())
            if proll.max() != 0:
                proll /= proll.max()
                if boolean:
                    proll = proll > 0
                    proll = proll.astype(int)
                proll *= 127
            proll = proll.astype(int)
            pianoroll_to_midi(
                proll, fs, program, filepath=filepath+'{}.midi'.format(i))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("globstr", type=str,
                        help="Glob string")
    parser.add_argument("fs", type=int, default=10,
                        help="Sampling rate per second")
    parser.add_argument("-p", "--program", type=int, default=1,
                        help="MIDI program")
    parser.add_argument("-t", "--threshold", type=lambda x: float(x), default=0,
                        help="Threshold for silence")
    parser.add_argument("-s", "--samples", type=int, default=10,
                        help="Number of samples per file")
    parser.add_argument("-b", "--boolean", type=int, default=0,
                        help="Ignore velocity")

    args = parser.parse_args()
    convert(args.globstr, args.fs, args.program, args.threshold, args.samples,
            args.boolean)
