#!/usr/bin/python

import argparse
import numpy as np
import glob2 as glob


def convert(globstr, samples):
    for filepath in glob.glob(globstr):
        prolls = np.load(filepath)
        print('{}, {}'.format(filepath, prolls.shape))
        text = '{}\n'.format(filepath)
        for i in range(min(samples, len(prolls))):
            proll = prolls[i]
            if len(proll.shape) == 3:
                proll = proll[0]
            cur_txt = ''.join([chr(x) for x in np.argmax(proll, axis=1)])
            text += '{}, {}\n'.format(i, cur_txt)
        with open(filepath+'.txt', "w") as text_file:
            text_file.write(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("globstr", type=str,
                        help="Glob string")
    parser.add_argument("-s", "--samples", type=int, default=10,
                        help="Number of samples per file")

    args = parser.parse_args()
    print(args)
    convert(args.globstr, args.samples)
