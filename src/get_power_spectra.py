#!/usr/bin/env python3

import sys, os
import numpy as np
import ase.io
import metatensor
from libs.config import read_config
from libs.lsoap import generate_lambda_soap_wrapper, make_rascal_hypers
from libs.functions import get_elements_list, print_progress
from libs.basis import basis_read
from libs.multi import multi_process

USEMPI=1


def main():
    o, p = read_config(sys.argv)

    def do_mol(imol):
        #if os.path.exists(f'{p.splitpsfilebase}_{imol}.npz'):
        #    return
        #print(imol)
        soap = generate_lambda_soap_wrapper(mols[imol], rascal_hypers, neighbor_species=elements,
                                            normalize=o.ps_normalize, min_norm=o.ps_min_norm,
                                            lmax=lmax)
        metatensor.save(f'{p.splitpsfilebase}_{imol}.mts', soap)

    rascal_hypers = make_rascal_hypers(o.soap_rcut, o.soap_ncut, o.soap_lcut, o.soap_sigma)

    lmax, _ = basis_read(p.basisfilename)
    mols = ase.io.read(p.xyzfilename, ":")
    elements = get_elements_list([mol.get_atomic_numbers() for mol in mols])

    print(f'{rascal_hypers=}')
    print(f'{elements=}')
    print(f'{lmax=}')
    print(f'{o.ps_min_norm=} {o.ps_normalize=}')

    nmol = len(mols)
    if USEMPI==0:
        for imol in range(nmol):
            print_progress(imol, nmol)
            do_mol(imol)
    else:
        multi_process(nmol, do_mol)


if __name__=='__main__':
    main()
