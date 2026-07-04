#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
from libs.config import read_config
from libs.functions import get_training_sets, Basis
from libs.get_matrices_A import get_a
from libs.get_matrices_B import get_b


def main():
    args, o, p = read_config(sys.argv, return_args=['get_b_matrix', 'mpi'])

    ref_elements = pd.read_csv(p.reference_environments)['q'].to_numpy()
    basis = Basis(o.basisname, elements=ref_elements)

    # training set selection
    ntrains, train_configs = get_training_sets(p.train_test_sets, o.fracs)
    ntrains = np.pad(ntrains, (0, 1), 'constant', constant_values=0)

    if args.get_b_matrix:
        bmatfiles = [p.bmat.format(train_frac=frac) for frac in o.fracs]
        get_b(basis, ref_elements, ntrains, train_configs,
              p.metric_matrix, p.kernel_nm, bmatfiles, use_mpi=args.mpi)
    else:
        avecfiles = [p.avec.format(train_frac=frac) for frac in o.fracs]
        get_a(basis, ref_elements, ntrains, train_configs,
              p.projection, p.kernel_nm, avecfiles)


if __name__=='__main__':
    main()
