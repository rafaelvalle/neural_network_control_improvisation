#!/usr/bin/python

import argparse
import numpy as np
from music_utils import pianoroll_to_midi
from data_processing import postprocess_proll, offset_proll
import glob2 as glob


def convert(globstr, fs, program, threshold, samples, boolean, argmax,
            offset, concat):
    for filepath in glob.glob(globstr):
        prolls = np.load(filepath)
        print('{}, {}'.format(filepath, prolls.shape))
        if len(prolls.shape) == 2:
            prolls = prolls.reshape((1,) + prolls.shape)

        for i in range(min(samples, len(prolls))):
            proll = prolls[i]
            if len(proll.shape) == 3:
                proll = proll[0]
            proll = offset_proll(proll, offset)
            postprocess_proll(proll, threshold, argmax, boolean)
            pianoroll_to_midi(
                proll, fs, program, filepath=filepath+'{}.midi'.format(i),
                threshold=concat)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    parser.add_argument("-m", "--argmax", type=int, default=0,
                        help="Argmax per timestep")
    parser.add_argument("-o", "--offset", type=int, default=0,
                        help="Piano roll pitch offset to start")
    parser.add_argument("-c", "--concat", type=int, default=np.inf,
                        help="Threshold for merging")

    args = parser.parse_args()
    print(args)
    convert(args.globstr, args.fs, args.program, args.threshold, args.samples,
            args.boolean, args.argmax, args.offset, args.concat)
