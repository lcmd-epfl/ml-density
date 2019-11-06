#!/usr/bin/python

import numpy as np
import time
import ase.io
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
xyzfile = ase.io.read(xyzfilename,":")
ndata = len(xyzfile)
#======================= system parameters
atomic_symbols = []
natoms = np.zeros(ndata,int)
for i in xrange(len(xyzfile)):
    atomic_symbols.append(xyzfile[i].get_chemical_symbols())
    natoms[i] = int(len(atomic_symbols[i]))

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

    i = 0
    for iat in xrange(natoms[iconf]):
        spe = atoms[iat]

        i += nmax[(spe,0)]

        if(lmax[spe]<1):
          continue

        for n1 in xrange(nmax[(spe,1)]):
            idx[i  ] = i+1
            idx[i+1] = i+2
            idx[i+2] = i
            i += 3

        for l in xrange(2, lmax[spe]+1):
            i += (2*l+1)*nmax[(spe,l)]

    idx = idx.tolist()

    #==================================================
    coeffs = np.loadtxt(coefffilebase+str(iconf)+".dat")
    overlap = np.load(overfilebase+str(iconf)+".npy")

    coef  = coeffs[idx]
    over1 = (np.transpose(overlap)) [idx]
    over  = (np.transpose(over1))   [idx]

    np.save(goodcoeffilebase+str(iconf)+".npy",coef)
    np.save(goodoverfilebase+str(iconf)+".npy",over)
    print "time =", time.time()-start

