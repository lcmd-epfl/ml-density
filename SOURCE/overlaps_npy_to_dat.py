#!/usr/bin/python

import numpy as np
import os
import sys
import ase.io
from config import Config

conf = Config()

xyzfilename      = conf.paths['xyzfile']
goodoverfilebase = conf.paths['goodover_base']
overdatbase      = conf.paths['over_dat_base']

xyzfile = ase.io.read(xyzfilename,":")
ndata = len(xyzfile)

for iconf in xrange(ndata):
    print "iconf = ", iconf
    # 7 times faster than numpy:
    cmd = os.path.dirname(sys.argv[0]) + "/npy2dat " + goodoverfilebase+str(iconf)+'.npy' + " " + overdatbase+str(iconf)+'.dat'
    os.system(cmd)

