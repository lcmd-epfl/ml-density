#!/usr/bin/env python3
"""Run PITC whole-molecule predictive-variance computation for train/test/extrapolation subsets."""

import numpy as np
import pandas as pd
import ase.io
from tqdm import tqdm
from libs.config import get_settings
from libs.functions import moldata_read, Basis, Subset, get_dataset_paths
from libs.variance_lib import compute_molecule_sigma, molecule_variance_trace, load_reference_metric, compute_reference_metric
from libs.logger_setup import setup_logger

logger = setup_logger(__name__, __file__)


def main():  # noqa: D103
    args, o, p = get_settings(return_args=['training', 'extra'])

    if not o.full_gpr:
        msg = 'variance.py requires full_gpr=True in the config (no PITC Cholesky factor exists otherwise)'
        raise RuntimeError(msg)

    dataset_paths = get_dataset_paths(p, extra=args.extra)
    ref_elements = pd.read_csv(p.reference_environments)['q'].to_numpy()
    basis = Basis(o.basisname, elements=ref_elements)
    frac_list = o.fracs[-1:] if args.extra else o.fracs

    if args.extra:
        asemols = ase.io.read(dataset_paths.xyz, ':')
        atomic_numbers = [mol.get_atomic_numbers() for mol in asemols]
        positions = [mol.get_positions() for mol in asemols]
        test_configs = np.arange(len(atomic_numbers))
        subset = 'extra'
    else:
        atomic_numbers = moldata_read(dataset_paths.xyz)
        subsets = Subset(p.train_test_sets)
        subset = 'training' if args.training else 'test'

    for frac in frac_list:
        l_factor = np.load(p.cholesky_pitc.format(train_frac=frac))

        if not args.extra:
            test_configs = subsets.get_training(frac) if args.training else subsets.get_test()

        logger.debug(f'Number of {subset} molecules = {len(test_configs)}')

        traces = []
        for imol in tqdm(test_configs):
            atoms = atomic_numbers[imol]
            if args.extra:
                s_star = compute_reference_metric(atoms, positions[imol], o.basisname)
            else:
                s_star = load_reference_metric(atoms, imol, o.basisname, p.metric_matrix)

            sigma_star = compute_molecule_sigma(basis, atoms, imol, ref_elements, l_factor,
                                                 dataset_paths.kernel, dataset_paths.power)
            traces.append((imol, molecule_variance_trace(sigma_star, s_star)))

        pd.DataFrame(traces, columns=['mol_idx', 'trace_variance']).to_csv(
            p.var_trace.format(subset=subset, train_frac=frac), index=False)


if __name__=='__main__':
    main()
