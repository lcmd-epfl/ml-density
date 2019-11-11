#!/usr/bin/python

import numpy as np
import ase.io
from config import Config

import ctypes
import os
import sys

npy2dat = ctypes.cdll.LoadLibrary(os.path.dirname(sys.argv[0])+"/npy2dat.so")
npy2dat.convert.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
npy2dat.convert.restype = ctypes.c_int

conf = Config()

xyzfilename      = conf.paths['xyzfile']
goodoverfilebase = conf.paths['goodover_base']
overdatbase      = conf.paths['over_dat_base']

xyzfile = ase.io.read(xyzfilename,":")
ndata = len(xyzfile)

for iconf in xrange(ndata):
    print "iconf = ", iconf
    # 7 times faster than numpy:
    ret = npy2dat.convert(goodoverfilebase+str(iconf)+'.npy', overdatbase+str(iconf)+'.dat')
    if ret:
        print "warning: returned", ret

