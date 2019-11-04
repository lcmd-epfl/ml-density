#!/usr/bin/python

import numpy as np
import ase.io
from config import Config

conf = Config()

def set_variable_values():
    seed  = conf.get_option('seed'        ,    1, int  )
    train = conf.get_option('train_size'  , 1000, int  )
    return [seed, train]

[seed, train] = set_variable_values()

xyzfilename      = conf.paths['xyzfile']
trainfilename    = conf.paths['trainingselfile']

xyzfile = ase.io.read(xyzfilename,":")
fullsize = len(xyzfile)

np.random.seed(seed)
output = np.random.choice(fullsize, train, replace=False)
output.sort()
np.savetxt(trainfilename, output, fmt='%i')

