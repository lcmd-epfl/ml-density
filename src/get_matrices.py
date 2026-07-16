#!/usr/bin/env python3
"""Assemble the regression Gram matrix and target vector (Python implementation)."""

import itertools
import pandas as pd
from libs.config import get_settings
from libs.functions import Basis, Subset, moldata_read
from libs.target_vector import get_target_vector
from libs.gram_matrix import get_gram_matrix
from libs.logger_setup import setup_logger

logger = setup_logger(__name__, __file__)


def main():  # noqa: D103
    args, o, p = get_settings(return_args=['get_gram_matrix', 'mpi'])

    if o.full_gpr and not args.get_gram_matrix:
        msg = 'get_matrices.py (without -b) does not support full_gpr -- run get_matrices.py -b instead, which builds both the target vector and Gram matrix together'
        raise RuntimeError(msg)

    ref_elements = pd.read_csv(p.reference_environments)['q'].to_numpy()
    basis = Basis(o.basisname, elements=ref_elements)

    # training set selection
    ntrains, train_configs = Subset(p.train_test_sets).get_training_all(o.fracs)
    ntrains = list(itertools.pairwise([0, *ntrains]))

    if args.get_gram_matrix:
        atomic_numbers = moldata_read(p.xyzfilename)
        get_gram_matrix(basis, ref_elements, o.fracs, ntrains, train_configs, p, o, atomic_numbers, use_mpi=args.mpi)
    else:
        get_target_vector(basis, ref_elements, o.fracs, ntrains, train_configs, p)


if __name__=='__main__':
    main()
