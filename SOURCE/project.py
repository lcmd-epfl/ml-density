#!/usr/bin/env python3

import sys
import numpy as np
import equistore
from config import Config, get_config_path
from basis import basis_read
from functions import moldata_read, print_progress, get_elements_list, nao_for_mol
from libs.tmap import averages2tmap, vector2tmap


def set_variable_values(conf):
    reorder_ao  = conf.get_option('reorder_ao'      ,  0, int)
    copy_metric = conf.get_option('copy_metric'     ,  1, int)
    return [reorder_ao, copy_metric]


def main():
    path = get_config_path(sys.argv)
    conf = Config(config_path=path)
    [reorder_ao, copy_metric] = set_variable_values(conf)
    xyzfilename      = conf.paths['xyzfile']
    basisfilename    = conf.paths['basisfile']
    coefffilebase    = conf.paths['coeff_base']
    overfilebase     = conf.paths['over_base']
    goodcoeffilebase = conf.paths['goodcoef_base']
    goodoverfilebase = conf.paths['goodover_base']
    avfile           = conf.paths['averages_file']
    baselinedwbase   = conf.paths['baselined_w_base']

    print(f'{copy_metric=}')
    print(f'{reorder_ao=}')

    (nmol, _, atomic_numbers) = moldata_read(xyzfilename)
    elements, counts = get_elements_list(atomic_numbers, return_counts=True)
    nenv = dict(zip(elements, counts))
    (el_dict, lmax, nmax) = basis_read(basisfilename)

    if list(elements) != list(el_dict.values()):
        print("different elements in the molecules and in the basis:", elements_in_set, "and", list(el_dict.values()) )
        exit(1)

    coefficients = []

    for imol, atoms in enumerate(atomic_numbers):
        print_progress(imol, nmol)
        idx = reorder_idx(atoms, lmax, nmax, reorder_ao)
        try:
            coef = np.loadtxt(f'{coefffilebase}{imol}.dat')
        except:
            coef = np.load(f'{coefffilebase}{imol}.npy')
        good_coef = coef[idx]
        coefficients.append(good_coef)
        np.save(f'{goodcoeffilebase}{imol}.npy', good_coef)
        if copy_metric:
            over      = np.load(f'{overfilebase}{imol}.npy')
            good_over = over[np.ix_(idx,idx)]
            np.save(f'{goodoverfilebase}{imol}.npy', good_over)

    av_coefs = get_averages(lmax, nmax, coefficients, atomic_numbers)

    for imol, (coef, atoms) in enumerate(zip(coefficients, atomic_numbers)):
        print_progress(imol, nmol)
        over  = np.load(f'{goodoverfilebase}{imol}.npy')
        i = 0
        for q in atoms:
            coef[i:i+nmax[(q,0)]] -= av_coefs[q]
            for l in range(lmax[q]+1):
                i += (2*l+1)*nmax[(q,l)]
        proj = over @ coef
        np.savetxt(f'{baselinedwbase}{imol}.dat', proj)
        proj_tmap = vector2tmap(atoms, lmax, nmax, proj)
        equistore.save(f'{baselinedwbase}{imol}.npz', proj_tmap)

    equistore.save(avfile, averages2tmap(av_coefs))


def reorder_idx(atoms, lmax, nmax, reorder_ao=False):
    nao = nao_for_mol(atoms, lmax, nmax)
    idx = np.arange(nao, dtype=int)
    if reorder_ao:
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


def get_averages(lmax, nmax, coefficients, atomic_numbers):
    elements, counts = get_elements_list(atomic_numbers, return_counts=True)
    nenv = dict(zip(elements, counts))
    av_coefs = {q: np.zeros(nmax[(q, 0)]) for q in elements}
    for coef, atoms in zip(coefficients, atomic_numbers):
        i = 0
        for q in atoms:
            av_coefs[q] += coef[i:i+nmax[(q,0)]]
            for l in range(lmax[q]+1):
                i += (2*l+1)*nmax[(q,l)]
    for q in elements:
        av_coefs[q] /= nenv[q]
    return av_coefs


if __name__=='__main__':
    main()
