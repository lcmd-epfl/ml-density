#!/usr/bin/env python3
"""Compute the reference-reference kernel matrix."""

import metatensor
from libs.functions import Basis
from libs.config import get_settings
from libs.kernels_lib import kernel_mm
from libs.logger_setup import setup_logger

logger = setup_logger(__name__, __file__)


def main():  # noqa: D103
    o, p = get_settings()

    power_ref = metatensor.load(p.reference_power_spectra)
    basis = Basis(o.basisname, elements=power_ref.keys.column('center_type'))

    k_MM = kernel_mm(basis.lmax, power_ref)
    metatensor.save(p.kernel_mm, k_MM)


if __name__=='__main__':
    main()
