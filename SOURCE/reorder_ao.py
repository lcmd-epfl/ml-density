#!/usr/bin/env python3

import numpy as np
from config import Config
from basis import basis_read
from functions import moldata_read,print_progress

conf = Config()

xyzfilename      = conf.paths['xyzfile']
basisfilename    = conf.paths['basisfile']
coefffilebase    = conf.paths['coeff_base']
overfilebase     = conf.paths['over_base']
goodcoeffilebase = conf.paths['goodcoef_base']
goodoverfilebase = conf.paths['goodover_base']


(nmol, natoms, atomic_numbers) = moldata_read(xyzfilename)

# elements dictionary, max. angular momenta, number of radial channels
(el_dict, lmax, nmax) = basis_read(basisfilename)

for imol in range(nmol):

    print_progress(imol, nmol)

    atoms = atomic_numbers[imol]
    #==================================================
    nao = 0
    for iat in range(natoms[imol]):
        for l in range(lmax[atoms[iat]]+1):
            nao += nmax[(atoms[iat],l)]*(2*l+1)

    idx = np.arange(nao, dtype=int)

    i = 0
    for iat in range(natoms[imol]):
        q = atoms[iat]

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

    idx = idx.tolist()

    #==================================================
    coef      = np.loadtxt(coefffilebase+str(imol)+".dat")
    good_coef = coef[idx]
    np.save(goodcoeffilebase+str(imol)+".npy", good_coef)

    over      = np.load(overfilebase+str(imol)+".npy")
    over1     = (np.transpose(over)) [idx]
    good_over = (np.transpose(over1))[idx]
    np.save(goodoverfilebase+str(imol)+".npy", good_over)

