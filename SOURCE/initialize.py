#!/usr/bin/python

import numpy as np
import time
import ase
from ase import io
from ase.io import read
from config import Config
from basis import basis_read

conf = Config()

xyzfilename      = conf.paths['xyzfile']
basisfilename    = conf.paths['basisfile']
coefffilebase    = conf.paths['coeff_base']
overfilebase     = conf.paths['over_base']
goodcoeffilebase = conf.paths['goodcoef_base']
goodoverfilebase = conf.paths['goodover_base']


#========================== system definition
xyzfile = read(xyzfilename,":")
ndata = len(xyzfile)
#======================= system parameters
atomic_symbols = []
natoms = np.zeros(ndata,int)
for i in xrange(len(xyzfile)):
    atomic_symbols.append(xyzfile[i].get_chemical_symbols())
    natoms[i] = int(len(atomic_symbols[i]))
natmax = max(natoms)

# species dictionary, max. angular momenta, number of radial channels
(spe_dict, lmax, nmax) = basis_read(basisfilename)

#===================================================== start decomposition
for iconf in xrange(ndata):
    start = time.time()
    print "-------------------------------"
    print "iconf = ", iconf
    atoms = atomic_symbols[iconf]
    #==================================================
    totsize = 0
    for iat in xrange(natoms[iconf]):
        for l in xrange(lmax[atoms[iat]]+1):
            totsize += nmax[(atoms[iat],l)]*(2*l+1)

    idx = np.arange(totsize, dtype=int)

    i1 = 0
    for iat in xrange(natoms[iconf]):
        spe1 = atoms[iat]

        i1 += nmax[(spe1,0)]

        if(lmax[spe1]<1):
          continue

        for n1 in xrange(nmax[(spe1,1)]):
            idx[i1  ] = i1+1
            idx[i1+1] = i1+2
            idx[i1+2] = i1
            i1 += 3

        for l1 in xrange(2, lmax[spe1]+1):
            i1 += (2*l1+1)*nmax[(spe1,l1)]

    idx = idx.tolist()

    #==================================================
    coeffs = np.loadtxt(coefffilebase+str(iconf)+".dat")
    overlap = np.load(overfilebase+str(iconf)+".npy")

    Coef  = coeffs[idx]
    Over1 = (np.transpose(overlap)) [idx]
    Over  = (np.transpose(Over1))   [idx]

    np.save(goodcoeffilebase+str(iconf)+".npy",Coef)
    np.save(goodoverfilebase+str(iconf)+".npy",Over)
    print "time =", time.time()-start

