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


(nmol, natoms, atomic_numbers) = moldata_read(xyzfilename)

# how many atoms of each element we have
nenv = {}
atomic_numbers_joined = [item for sublist in atomic_numbers for item in sublist]
elements_in_set = list(set(atomic_numbers_joined))
elements_in_set.sort()
for q in elements_in_set:
    nenv[q] = atomic_numbers_joined.count(q)
    print(chemical_symbols[q], nenv[q])

# elements dictionary, max. angular momenta, number of radial channels
(el_dict, lmax, nmax) = basis_read(basisfilename)

if elements_in_set != list(el_dict.values()):
  print("different elements in the molecules and in the basis:", elements_in_set, "and", list(el_dict.values()) )
  exit(1)

av_coefs = {}
for q in el_dict.values():
    av_coefs[q] = np.zeros(nmax[(q,0)],float)

print("computing averages:")
for imol in range(nmol):
    print_progress(imol, nmol)
    atoms = atomic_numbers[imol]
    coef = np.load(goodcoeffilebase+str(imol)+".npy")
    i = 0
    for iat in range(natoms[imol]):
        q = atoms[iat]
        for n in range(nmax[(q,0)]):
            av_coefs[q][n] += coef[i]
            i += 1
        for l in range(1,lmax[q]+1):
            i += (2*l+1)*nmax[(q,l)]
for q in el_dict.values():
    av_coefs[q] /= nenv[q]
    np.save(avdir+chemical_symbols[q]+".npy",av_coefs[q])

print()
print("computing baselined projections:")
for imol in range(nmol):
    print_progress(imol, nmol)
    atoms = atomic_numbers[imol]

    coef = np.load(goodcoeffilebase+str(imol)+".npy")
    over = np.load(goodoverfilebase+str(imol)+".npy")

    i = 0
    for iat in range(natoms[imol]):
        q = atoms[iat]
        for n in range(nmax[(q,0)]):
            coef[i] -= av_coefs[q][n]
            i += 1
        for l in range(1,lmax[q]+1):
            i += (2*l+1)*nmax[(q,l)]

    proj = np.dot(over,coef)
    np.savetxt(baselinedwbase+str(imol)+".dat",proj, fmt='%.10e')

