#!/usr/bin/env python3
"""Run model prediction and export coefficients for main or extra datasets."""

from functools import partial
import numpy as np
import metatensor
from tqdm import tqdm
from qstack.io.metatensor import join
from qstack import reorder
from libs.config import get_settings
from libs.functions import moldata_read, Basis, Subset, make_dummy_mol, get_dataset_paths
from libs.predict import run_prediction
from libs.tmap import tmap2vector, tmap_add
from libs.logger_setup import setup_logger

logger = setup_logger(__name__, __file__)


def main():  # noqa: D103
    args, o, p = get_settings(return_args=['training', 'extra'])
    dataset_paths = get_dataset_paths(p, extra=args.extra)
    frac_list = o.fracs[-1:] if args.extra else o.fracs

    atomic_numbers = moldata_read(dataset_paths.xyz)
    averages = metatensor.load(p.spherical_averages)
    basis = Basis(o.basisname, elements=averages.keys.column('center_type'))
    subsets = Subset(p.train_test_sets)

    for frac in frac_list:
        weights = metatensor.load(p.weights.format(train_frac=frac))
        pred_configs, pred_mols, pred_path, c_path_fmter = _get_split(args, p, atomic_numbers, subsets, frac)
        predictions = run_prediction(pred_configs, pred_mols, basis, weights, dataset_paths.kernel,
                                     averages=averages if args.extra else None)
        if pred_path:
            metatensor.save(pred_path, join(predictions))

        for imol, atoms, pred in zip(tqdm(pred_configs), pred_mols, predictions, strict=True):
            if not args.extra:
                tmap_add(pred, averages)
            c = tmap2vector(atoms, basis.llist, pred)
            if o.output_coeff_order != 'gpr':
                pyscf_mol = make_dummy_mol(atoms, basis=o.basisname, ignore=True)
                c = reorder.reorder_ao(pyscf_mol, c, dest=o.output_coeff_order, src='gpr')
            np.savetxt(c_path_fmter(order=o.output_coeff_order, imol=imol), c)


def _get_split(args, p, atomic_numbers, subsets, frac):
    if args.extra:
        pred_configs = np.arange(len(atomic_numbers))
        pred_path = None
        c_path_fmter = p.extra_predicted_coeff.format
    else:
        pred_configs, pred_path = subsets.get_pred_idx(args.training, p.predictions, frac)
        c_path_fmter = partial(p.predicted_coeff.format, train_frac=frac)
    return pred_configs, atomic_numbers[pred_configs], pred_path, c_path_fmter


if __name__=='__main__':
    main()
