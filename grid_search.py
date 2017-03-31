#!/usr/bin/python
import subprocess
import itertools
import numpy as np

lrs = np.arange(1e-5, 1e-3, 1e-5)
cl_iters = [100, 250, 500]
for lr, cl_iter in itertools.product(lrs, cl_iters): 
    cmd = "./wcgan_text.py 'text' 'cont-enc' 'cont-enc' --clr {0} --glr {0} -n 20 -e 100 -m 32 -c 0 --cbn 1 --gbn 1 --clip 0.01 -l 'wgan' -z 100 --cl_iters {1} --cl_freq 100 --name 'text_lr_{0}_cliters_{1}'".format(lr, cl_iter)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()
    p_status = p.wait()
    with open("text_lr_{0}_cliters_{1}".format(lr, cl_iter), "w") as fp:
        fp.write(output)
