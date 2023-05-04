#!/usr/bin/env python3

import sys
import equistore
import numpy as np
from config import read_config
import ase.io
from soap.lsoap import generate_lambda_soap_wrapper, remove_high_l
from functions import get_elements_list, print_progress
from basis import basis_read
from libs.multi import multi_process

USEMPI=1


def main():
    o, p = read_config(sys.argv)

    def do_mol(imol):
        soap = generate_lambda_soap_wrapper(mols[imol], rascal_hypers, neighbor_species=elements, normalize=True)
        soap = remove_high_l(soap, lmax)
        equistore.save(f'{p.splitpsfilebase}_{imol}.npz', soap)

    rascal_hypers = {
        "cutoff": o.soap_rcut,
        "max_radial": o.soap_ncut,
        "max_angular": o.soap_lcut,
        "atomic_gaussian_width": o.soap_sigma,
        "radial_basis": {"Gto": {}},
        "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
        "center_atom_weight": 1.0,
    }

    _, lmax, _ = basis_read(p.basisfilename)
    mols = ase.io.read(p.xyzfilename, ":")
    elements = get_elements_list([mol.get_atomic_numbers() for mol in mols])

    print(f'{rascal_hypers=}')
    print(f'{elements=}')
    print(f'{lmax=}')

    nmol = len(mols)
    if USEMPI==0:
        for imol in range(nmol):
            print_progress(imol, nmol)
            do_mol(imol)
    else:
        multi_process(nmol, do_mol)


if __name__=='__main__':
    main()
