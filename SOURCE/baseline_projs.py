#!/usr/bin/python3

import numpy as np
from config import Config
from basis import basis_read
from ase.data import chemical_symbols
from functions import moldata_read,print_progress

conf = Config()

xyzfilename      = conf.paths['xyzfile']
basisfilename    = conf.paths['basisfile']
goodcoeffilebase = conf.paths['goodcoef_base']
goodoverfilebase = conf.paths['goodover_base']
avdir            = conf.paths['averages_dir']
baselinedwbase   = conf.paths['baselined_w_base']


(ndata, natoms, atomic_numbers) = moldata_read(xyzfilename)

# how many atoms of each element we have
nenv = {}
atomic_numbers_joined = [item for sublist in atomic_numbers for item in sublist]
elements_in_set = list(set(atomic_numbers_joined))
for q in elements_in_set:
    nenv[q] = atomic_numbers_joined.count(q)
    print(chemical_symbols[q], nenv[q])

# species dictionary, max. angular momenta, number of radial channels
(spe_dict, lmax, nmax) = basis_read(basisfilename)

av_coefs = {}
for spe in spe_dict.values():
    av_coefs[spe] = np.zeros(nmax[(spe,0)],float)

print("computing averages:")
for iconf in range(ndata):
    print_progress(iconf, ndata)
    atoms = atomic_numbers[iconf]
    coef = np.load(goodcoeffilebase+str(iconf)+".npy")
    i = 0
    for iat in range(natoms[iconf]):
        spe = atoms[iat]
        for n in range(nmax[(spe,0)]):
            av_coefs[spe][n] += coef[i]
            i += 1
        for l in range(1,lmax[spe]+1):
            i += (2*l+1)*nmax[(spe,l)]
for spe in spe_dict.values():
    av_coefs[spe] /= nenv[spe]
    np.save(avdir+chemical_symbols[spe]+".npy",av_coefs[spe])

print()
print("computing baselined projections:")
for iconf in range(ndata):
    print_progress(iconf, ndata)
    atoms = atomic_numbers[iconf]
    #==================================================
    coef = np.load(goodcoeffilebase+str(iconf)+".npy")
    over = np.load(goodoverfilebase+str(iconf)+".npy")
    #==================================================
    i = 0
    for iat in range(natoms[iconf]):
        spe = atoms[iat]
        for n in range(nmax[(spe,0)]):
            coef[i] -= av_coefs[spe][n]
            i += 1
        for l in range(1,lmax[spe]+1):
            i += (2*l+1)*nmax[(spe,l)]

    proj = np.dot(over,coef)
    np.savetxt(baselinedwbase+str(iconf)+".dat",proj, fmt='%.10e')

