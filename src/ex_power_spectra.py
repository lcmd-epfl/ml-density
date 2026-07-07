#!/usr/bin/env python3
"""Compute power spectra for out-of-sample molecules."""

import ase.io
import metatensor
from tqdm import tqdm
from libs.config import get_settings
from libs.lsoap import generate_lambda_soap_wrapper, make_rascal_hypers
from libs.functions import get_elements, moldata_read, Basis
from libs.logger_setup import setup_logger

logger = setup_logger(__name__, __file__)


def main():  # noqa: D103
    o, p = get_settings()

    mols_ex = ase.io.read(p.xyzexfilename, ":")
    atomic_numbers = moldata_read(p.xyzfilename)

    elements = get_elements(atomic_numbers)
    elements_ex = get_elements(mols_ex)
    if not set(elements_ex).issubset(elements):
        msg = f'Different elements in the molecule and in the training set: {elements_ex} and {elements}'
        raise RuntimeError(msg)

    basis = Basis(o.basisname, elements=elements_ex)
    rascal_hypers = make_rascal_hypers(o.soap_rcut, o.soap_ncut, o.soap_lcut, o.soap_sigma)

    logger.debug(f'rascal_hypers={rascal_hypers}')
    logger.debug(f'elements={elements}')
    logger.debug(f'basis.lmax={basis.lmax}')
    logger.debug(f'ps_min_norm={o.ps_min_norm} ps_normalize={o.ps_normalize}')

    for imol, mol in enumerate(tqdm(mols_ex)):
        soap = generate_lambda_soap_wrapper(mol, rascal_hypers, neighbor_species=elements,
                                            normalize=o.ps_normalize, min_norm=o.ps_min_norm,
                                            lmax=basis.lmax)
        metatensor.save(p.extra_power_spectrum.format(imol), soap)


if __name__=='__main__':
    main()
