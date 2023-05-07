#!/usr/bin/env python3

import sys
import numpy as np
import ase.io
import equistore
from libs.config import read_config
from libs.lsoap import generate_lambda_soap_wrapper, remove_high_l, make_rascal_hypers
from libs.functions import get_elements_list, print_progress, moldata_read
from libs.basis import basis_read

def main():
    o, p = read_config(sys.argv)

    mols_ex = ase.io.read(p.xyzexfilename, ":")
    atomic_numbers = moldata_read(p.xyzfilename)

    elements = get_elements_list(atomic_numbers)
    elements_ex = get_elements_list([mol.get_atomic_numbers() for mol in mols_ex])
    if not set(elements_ex).issubset(elements):
        print(f'Different elements in the molecule and in the training set: {elements_ex} and {elements}')
        exit(1)

    lmax, _ = basis_read(p.basisfilename)
    rascal_hypers = make_rascal_hypers(o.soap_rcut, o.soap_ncut, o.soap_lcut, o.soap_sigma)

    print(f'{rascal_hypers=}')
    print(f'{elements=}')
    print(f'{lmax=}')
    print(f'{o.ps_min_norm=}')

    for imol, mol in enumerate(mols_ex):
        print_progress(imol, len(mols_ex))
        soap = generate_lambda_soap_wrapper(mol, rascal_hypers, neighbor_species=elements, normalize=True, min_norm=o.ps_min_norm)
        soap = remove_high_l(soap, lmax)
        equistore.save(f'{p.powerexbase}_{imol}.npz', soap)


if __name__=='__main__':
    main()
