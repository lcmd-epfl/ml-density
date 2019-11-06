#!/usr/bin/python

import numpy as np
import ase.io
from config import Config
from basis import basis_read

conf = Config()

xyzfilename      = conf.paths['xyzfile']
basisfilename    = conf.paths['basisfile']
goodcoeffilebase = conf.paths['goodcoef_base']
goodoverfilebase = conf.paths['goodover_base']
avdir            = conf.paths['averages_dir']
baselinedwbase   = conf.paths['baselined_w_base']


#========================== system definition
xyzfile = ase.io.read(xyzfilename,":")
ndata = len(xyzfile)
#======================= system parameters
atomic_symbols = []
natoms = np.zeros(ndata,int)
for i in xrange(len(xyzfile)):
    atomic_symbols.append(xyzfile[i].get_chemical_symbols())
    natoms[i] = len(atomic_symbols[i])

# how many atoms of each element we have
nenv = {}
atomic_symbols_joined = [item for sublist in atomic_symbols for item in sublist]
elements_in_set = list(set(atomic_symbols_joined))
for q in elements_in_set:
    nenv[q] = atomic_symbols_joined.count(q)
    print q, nenv[q]

# species dictionary, max. angular momenta, number of radial channels
(spe_dict, lmax, nmax) = basis_read(basisfilename)


av_coefs = {}
for spe in spe_dict.values():
    av_coefs[spe] = np.zeros(nmax[(spe,0)],float)

print "computing averages..."
for iconf in xrange(ndata):
    print "iconf = ", iconf
    atoms = atomic_symbols[iconf]
    coef = np.load(goodcoeffilebase+str(iconf)+".npy")
    i = 0
    for iat in xrange(natoms[iconf]):
        spe = atoms[iat]
        for n in xrange(nmax[(spe,0)]):
            av_coefs[spe][n] += coef[i]
            i += 1
        for l in xrange(1,lmax[spe]+1):
            i += (2*l+1)*nmax[(spe,l)]

print "saving averages..."
for spe in spe_dict.values():
    av_coefs[spe] /= nenv[spe]
    np.save(avdir+str(spe)+".npy",av_coefs[spe])


print "computing baselined projections..."
for iconf in xrange(ndata):
    print "iconf = ", iconf
    atoms = atomic_symbols[iconf]
    #==================================================
    coef = np.load(goodcoeffilebase+str(iconf)+".npy")
    over = np.load(goodoverfilebase+str(iconf)+".npy")
    #==================================================
    i = 0
    for iat in xrange(natoms[iconf]):
        spe = atoms[iat]
        for n in xrange(nmax[(spe,0)]):
            coef[i] -= av_coefs[spe][n]
            i += 1
        for l in xrange(1,lmax[spe]+1):
            i += (2*l+1)*nmax[(spe,l)]

    proj = np.dot(over,coef)
    np.savetxt(baselinedwbase+str(iconf)+".dat",proj, fmt='%.10e')

