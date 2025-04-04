#!/usr/bin/env python3

import sys
import numpy as np
from config import Config,get_config_path
import ase.io

path = get_config_path(sys.argv)
conf = Config(config_path=path)

def set_variable_values():
    seed  = conf.get_option('seed'        ,    1, int  )
    train = conf.get_option('train_size'  , 1000, int  )
    return [seed, train]

[seed, train] = set_variable_values()

xyzfilename      = conf.paths['xyzfile']
trainfilename    = conf.paths['trainingselfile']

xyzfile = ase.io.read(xyzfilename,":")
nmol = len(xyzfile)

np.random.seed(seed)
output = np.random.choice(nmol, train, replace=False)
np.savetxt(trainfilename, output, fmt='%i')

