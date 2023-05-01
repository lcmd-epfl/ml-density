#!/usr/bin/env python3

import sys
import numpy as np
import equistore
from config import Config,get_config_path
from basis import basis_read
from functions import moldata_read, print_progress, get_elements_list
from libs.tmap import averages2tmap


def main():
    path = get_config_path(sys.argv)
    conf = Config(config_path=path)
    xyzfilename      = conf.paths['xyzfile']
    basisfilename    = conf.paths['basisfile']
    goodcoeffilebase = conf.paths['goodcoef_base']
    goodoverfilebase = conf.paths['goodover_base']
    avfile           = conf.paths['averages_file']
    baselinedwbase   = conf.paths['baselined_w_base']


    (nmol, natoms, atomic_numbers) = moldata_read(xyzfilename)
    elements, counts = get_elements_list(atomic_numbers, return_counts=True)
    nenv = dict(zip(elements, counts))
    (el_dict, lmax, nmax) = basis_read(basisfilename)

    if list(elements) != list(el_dict.values()):
      print("different elements in the molecules and in the basis:", elements_in_set, "and", list(el_dict.values()) )
      exit(1)

    av_coefs = {q: np.zeros(nmax[(q, 0)]) for q in elements}

    for imol in range(nmol):
        coef = np.load(f'{goodcoeffilebase}{imol}.npy')
        atoms = atomic_numbers[imol]
        i = 0
        for q in atoms:
            av_coefs[q] += coef[i:i+nmax[(q,0)]]
            for l in range(lmax[q]+1):
                i += (2*l+1)*nmax[(q,l)]
    for q in el_dict.values():
        av_coefs[q] /= nenv[q]

    for imol in range(nmol):
        print_progress(imol, nmol)
        atoms = atomic_numbers[imol]
        coef = np.load(f'{goodcoeffilebase}{imol}.npy')
        over = np.load(f'{goodoverfilebase}{imol}.npy')
        i = 0
        for q in atoms:
            coef[i:i+nmax[(q,0)]] -= av_coefs[q]
            for l in range(lmax[q]+1):
                i += (2*l+1)*nmax[(q,l)]
        proj = over @ coef
        np.savetxt(f'{baselinedwbase}{imol}.dat', proj)

    equistore.save(avfile, averages2tmap(av_coefs))


if __name__=='__main__':
    main()
