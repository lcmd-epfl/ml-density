#!/usr/bin/env python3

import sys
import numpy as np
from config import Config
from basis import basis_read
from functions import moldata_read,print_progress,nao_for_mol

conf = Config()

xyzfilename      = conf.paths['xyzfile']
basisfilename    = conf.paths['basisfile']
coefffilebase    = conf.paths['coeff_base']
overfilebase     = conf.paths['over_base']
goodcoeffilebase = conf.paths['goodcoef_base']
goodoverfilebase = conf.paths['goodover_base']


only_c = False
reorder = True
if 'c' in sys.argv[1:]:
  only_c = True
if 'noreorder' in sys.argv[1:]:
  reorder = False

(nmol, natoms, atomic_numbers) = moldata_read(xyzfilename)

# elements dictionary, max. angular momenta, number of radial channels
(el_dict, lmax, nmax) = basis_read(basisfilename)

def reorder_idx(nao, atoms, lmax, nmax):
    idx = np.arange(nao, dtype=int)
    i = 0
    for q in atoms:
        i += nmax[(q,0)]
        if(lmax[q]<1):
            continue
        for n in range(nmax[(q,1)]):
            idx[i  ] = i+1
            idx[i+1] = i+2
            idx[i+2] = i
            i += 3
        for l in range(2, lmax[q]+1):
            i += (2*l+1)*nmax[(q,l)]
    return idx

for imol in range(nmol):

    print_progress(imol, nmol)

    atoms = atomic_numbers[imol]
    nao = nao_for_mol(atoms, lmax, nmax)

    if(reorder):
        idx = reorder_idx(nao, atoms, lmax, nmax)
    else:
        idx = np.arange(nao, dtype=int)

    coef      = np.loadtxt(coefffilebase+str(imol)+".dat")
    good_coef = coef[idx]
    np.save(goodcoeffilebase+str(imol)+".npy", good_coef)

    if only_c:
        continue
    over      = np.load(overfilebase+str(imol)+".npy")
    good_over = over[:,idx][idx,:]
    np.save(goodoverfilebase+str(imol)+".npy", good_over)

