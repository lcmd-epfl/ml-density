#!/usr/bin/python

import numpy as np
import ase
from ase import io
from ase.io import read
from config import Config

conf = Config()

xyzfilename      = conf.paths['xyzfile']
goodoverfilebase = conf.paths['goodover_base']
overdatbase      = conf.paths['over_dat_base']

xyzfile = read(xyzfilename,":")
ndata = len(xyzfile)

for iconf in xrange(ndata):
    print "iconf = ", iconf
    Over = np.load(goodoverfilebase+str(iconf)+".npy")
    np.savetxt(overdatbase+str(iconf)+".dat", np.concatenate(Over), fmt='%.10e')

