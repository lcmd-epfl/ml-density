#!/usr/bin/python3

import numpy as np
import ase.io
from config import Config
from functions import print_progress

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

for iconf in range(ndata):
    print_progress(iconf, ndata)
    fname1 = goodoverfilebase+str(iconf)+'.npy'
    fname2 = overdatbase+str(iconf)+'.dat'
    ret = npy2dat.convert(fname1.encode('ascii'), fname2.encode('ascii'))
    if ret:
        print("warning: returned", ret)

