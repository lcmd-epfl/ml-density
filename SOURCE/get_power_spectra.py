#!/usr/bin/env python3

import sys
import equistore
import numpy as np
from config import Config, get_config_path
import ase.io
from soap.lsoap import generate_lambda_soap_wrapper, remove_high_l
from functions import get_elements_list, print_progress
from basis import basis_read


def set_variable_values(conf):
    soap_sigma = conf.get_option('soap_sigma'  ,  0.3, float  )
    soap_rcut  = conf.get_option('soap_rcut '  ,  4.0, float  )
    soap_ncut  = conf.get_option('soap_ncut '  ,  8  , int    )
    soap_lcut  = conf.get_option('soap_lcut '  ,  6  , int    )
    return [soap_sigma, soap_rcut, soap_ncut, soap_lcut]


def main():
    path = get_config_path(sys.argv)
    conf = Config(config_path=path)
    [soap_sigma, soap_rcut, soap_ncut, soap_lcut] = set_variable_values(conf)
    xyzfilename     = conf.paths['xyzfile']
    basisfilename   = conf.paths['basisfile']
    splitpsfilebase = conf.paths['ps_split_base']

    rascal_hypers = {
        "cutoff": soap_rcut,
        "max_radial": soap_ncut,
        "max_angular": soap_lcut,
        "atomic_gaussian_width": soap_sigma,
        "radial_basis": {"Gto": {}},
        "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
        "center_atom_weight": 1.0,
    }

    _, lmax, _ = basis_read(basisfilename)
    mols = ase.io.read(xyzfilename, ":")
    elements = get_elements_list([mol.get_atomic_numbers() for mol in mols])

    print(f'{rascal_hypers=}')
    print(f'{elements=}')
    print(f'{lmax=}')

    for imol, mol in enumerate(mols):
        print_progress(imol, len(mols))
        soap = generate_lambda_soap_wrapper(mol, rascal_hypers, neighbor_species=elements, normalize=True)
        soap = remove_high_l(soap, lmax)
        equistore.save(f'{splitpsfilebase}_{imol}.npz', soap)


if __name__=='__main__':
    main()
