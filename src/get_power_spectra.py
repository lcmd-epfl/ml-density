#!/usr/bin/env python3

import os
import sys
import ase.io
import metatensor
from tqdm import trange
from libs.config import read_config
from libs.lsoap import generate_lambda_soap_wrapper, make_rascal_hypers
from libs.functions import get_elements, Basis
from libs.multi import multi_process


def main():
    args, o, p = read_config(sys.argv, return_args=['missing_only', 'mpi'])

    def do_mol(imol):
        ppath = p.power_spectrum.format(imol)
        if args.missing_only and os.path.exists(ppath):
            return
        soap = generate_lambda_soap_wrapper(mols[imol], rascal_hypers, neighbor_species=elements,
                                            normalize=o.ps_normalize, min_norm=o.ps_min_norm,
                                            lmax=basis.lmax)
        metatensor.save(ppath, soap)

    rascal_hypers = make_rascal_hypers(o.soap_rcut, o.soap_ncut, o.soap_lcut, o.soap_sigma)

    mols = ase.io.read(p.xyzfilename, ":")
    elements = get_elements(mols)
    basis = Basis(o.basisname, elements)

    print(f'{rascal_hypers=}')
    print(f'{elements=}')
    print(f'{basis.lmax=}')
    print(f'{o.ps_min_norm=} {o.ps_normalize=}')

    nmol = len(mols)
    if args.mpi:
        multi_process(nmol, do_mol)
    else:
        for imol in trange(nmol):
            do_mol(imol)


if __name__=='__main__':
    main()
