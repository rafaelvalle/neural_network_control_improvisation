#!/usr/bin/python

import argparse
import numpy as np
import glob2 as glob


def convert(globstr, samples):
    for filepath in glob.glob(globstr):
        encoded_texts = np.load(filepath)
        print('{}, {}'.format(filepath, encoded_texts.shape))
        text = '{}\n'.format(filepath)
        for i in range(min(samples, len(encoded_texts))):
            encoded_text = encoded_texts[i]
            if len(encoded_text.shape) == 3:
                encoded_text = encoded_text[0]
            cur_txt = ''.join([chr(x) for x in np.argmax(encoded_text, axis=0)])
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
