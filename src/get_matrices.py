#!/usr/bin/env python3
"""Assemble regression "B matrix" and "A vector" (Python implementation)."""

import itertools
import pandas as pd
from libs.config import get_settings
from libs.functions import Basis, Subset
from libs.get_matrices_A import get_a
from libs.get_matrices_B import get_b
from libs.logger_setup import setup_logger

logger = setup_logger(__name__, __file__)


def main():  # noqa: D103
    args, o, p = get_settings(return_args=['get_b_matrix', 'mpi'])

    ref_elements = pd.read_csv(p.reference_environments)['q'].to_numpy()
    basis = Basis(o.basisname, elements=ref_elements)

    # training set selection
    ntrains, train_configs = Subset(p.train_test_sets).get_training_all(o.fracs)
    ntrains = list(itertools.pairwise([0, *ntrains]))

    if args.get_b_matrix:
        get_b(basis, ref_elements, o.fracs, ntrains, train_configs, p, use_mpi=args.mpi)
    else:
        get_a(basis, ref_elements, o.fracs, ntrains, train_configs, p)


if __name__=='__main__':
    main()
