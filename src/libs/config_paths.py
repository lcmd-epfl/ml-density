"""Path and option specification assembly for project configuration."""

import sys
from types import SimpleNamespace
from collections import ChainMap
import logging
import numpy as np
from .config_parser import Config
from .config_utils import CheckFile, WhenMissing, PathSpecs, OptSpecs, defaults, Floats, Bool, Choice

logger = logging.getLogger('__main__')


def read_config(config_path=defaults.config, *, print_help=False):
    """Read configuration options and derive runtime path templates.

    Args:
        config_path (str): Path to the configuration file.
        print_help (bool): Print help and exit.

    Returns:
        tuple[types.SimpleNamespace, types.SimpleNamespace]: Parsed options namespace and resolved paths namespace.
    """
    def set_options():
        """Build option specifications by configuration section.

        Returns:
            dict[str, dict[str, OptSpecs]]: Specification of option groups and entries.
        """
        return {
                'options.training': {
                    'M'                  : OptSpecs('reference_environments', 100, int, 'Number of reference environments to select for sparse regression.'),
                    'seed'               : OptSpecs('seed', 1, int, 'Random seed for train/test splitting.'),
                    'train'              : OptSpecs('train_size', 1000, int, 'Number of molecules to assign to the training subset.'),
                    'fracs'              : OptSpecs('train_fractions', np.array([1.0]), Floats(), 'Comma-separated training fractions for learning curve.'),
                    'reg'                : OptSpecs('regularisation', 1e-6, float, 'Ridge regularization strength for regression.'),
                    'jit'                : OptSpecs('jitter', 1e-10, float, 'Diagonal regularization strength for regression.'),
                    'full_gpr'           : OptSpecs('full_gpr', default=False, dtype=Bool(), help='Enable full/exact GPR via PITC sparcification, unlocking access to the variance.'),
                    },
                'options.soap': {
                    'soap_sigma'         : OptSpecs('soap_sigma', 0.3, float, 'Gaussian width of atomic neighbor densities in λ-SOAP power spectra.'),
                    'soap_rcut'          : OptSpecs('soap_rcut', 4.0, float, 'Cutoff radius for λ-SOAP power spectra.'),
                    'soap_ncut'          : OptSpecs('soap_ncut', 8, int, 'Maximum radial basis index for λ-SOAP power spectra.'),
                    'soap_lcut'          : OptSpecs('soap_lcut', 6, int, 'Maximum angular momentum λ for λ-SOAP power spectra.'),
                    'ps_normalize'       : OptSpecs('power_spectra_normalize', default=True, dtype=Bool(), help='Enable normalization of λ-SOAP power spectra.'),
                    'ps_min_norm'        : OptSpecs('power_spectra_min_norm', 1e-20, float, 'Minimum norm threshold in λ-SOAP power spectra normalization.'),
                    },
                'options.rho': {
                    'process_metric'     : OptSpecs('process_metric', default=True, dtype=Bool(), help='If true, process Coulomb metric matrices from input files.'),
                    'use_charges'        : OptSpecs('number_of_electrons', 'none', Choice(str, ['none', 'charge', 'N'], name='number_of_electrons'), 'Column in the dataset CSV for target electron count.'),
                    'basisname'          : OptSpecs('basis', 'cc-pvqz-jkfit', str, 'Basis set name used to build AO representation.'),
                    'coeff_order'        : OptSpecs('coeff_order', 'pyscf', Choice(str, ['pyscf', 'gpr'], name='coeff_order', strict=False), 'AO ordering convention of input coefficient files.'),
                    'metric_order'       : OptSpecs('metric_order', 'pyscf', Choice(str, ['pyscf', 'gpr'], name='metric_order', strict=False), 'AO ordering convention of input metric matrices.'),
                    'output_coeff_order' : OptSpecs('output_coeff_order', 'gpr', Choice(str, ['pyscf', 'gpr'], name='output_coeff_order', strict=False), 'AO ordering convention for exported predicted coefficients.'),
                    },
                }

    def set_paths():
        """Build path specifications by configuration section.

        Returns:
            dict[str, dict[str, PathSpecs]]: Specification of path groups and entries.
        """
        return {
                'paths.input': {
                    'dataset'            : PathSpecs('dataset', WhenMissing.ERROR, CheckFile.ERROR_FILE, None, 'CSV dataset with molecule IDs and optional charge columns.'),
                    'xyz'                : PathSpecs('xyz', WhenMissing.ERROR, CheckFile.ERROR_DIR, None, 'Template path to input XYZ files (uses {mol_name}).'),
                    'input_metrics'      : PathSpecs('metrics', WhenMissing.ERROR, CheckFile.ERROR_DIR, None, 'Template path to input metric matrices.'),
                    'input_coeffs'       : PathSpecs('coeffs', WhenMissing.ERROR, CheckFile.ERROR_DIR, None, 'Template path to input coefficient files.'),
                    },
                'paths.output': {
                    '_weightsfilebase'   : PathSpecs('weights_base', WhenMissing.IGNORE, CheckFile.MAKE_DIR, 'INNER/weights', 'Base path for trained model weights.'),
                    '_predictfilebase'   : PathSpecs('predict_base', WhenMissing.IGNORE, CheckFile.MAKE_DIR, 'INNER/prediction', 'Base path for predicted TensorMap coefficients.'),
                    '_outfilebase'       : PathSpecs('output_base', WhenMissing.WARN, CheckFile.MAKE_DIR, 'INNER/predicted/rho', 'Base path for exported coefficient text files.'),
                    },
                'paths.extrapolation': {
                    'extra_xyzfilename'  : PathSpecs('xyz', WhenMissing.WARN, CheckFile.ERROR_FILE, None, 'XYZ file for extrapolation/out-of-sample molecules.'),
                    '_powerexbase'       : PathSpecs('power_spectra_base', WhenMissing.IGNORE, CheckFile.MAKE_DIR, 'INNER/extra/PS', 'Base path for extrapolation/OOS power spectra.'),
                    '_kernelexbase'      : PathSpecs('kernel_base', WhenMissing.IGNORE, CheckFile.MAKE_DIR, 'INNER/extra/kernel', 'Base path for extrapolation/OOS kernels.'),
                    '_outexfilebase'     : PathSpecs('output_base', WhenMissing.WARN, CheckFile.MAKE_DIR, 'INNER/extra/rho', 'Base path for extrapolation/OOS exported coefficients.'),
                    },
                'paths.internal': {
                    'xyzfilename'        : PathSpecs('xyz', WhenMissing.IGNORE, CheckFile.MAKE_DIR, 'INNER/dataset.xyz', 'Combined XYZ file written from all molecules.'),
                    '_splitpsfilebase'   : PathSpecs('power_spectra_split_base', WhenMissing.IGNORE, CheckFile.MAKE_DIR, 'INNER/PS/PS', 'Base path for per-molecule power spectra.'),
                    '_refsselfilebase'   : PathSpecs('reference_selection_base', WhenMissing.IGNORE, CheckFile.MAKE_DIR, 'INNER/SELECTIONS/refs_selection', 'Base path for selected reference environments.'),
                    '_powerrefbase'      : PathSpecs('power_spectra_reference_base', WhenMissing.IGNORE, CheckFile.MAKE_DIR, 'INNER/PS', 'Base path for merged reference power spectra.'),
                    '_kmmbase'           : PathSpecs('kmm_base', WhenMissing.IGNORE, CheckFile.MAKE_DIR, 'INNER/KMM', 'Base path for reference-reference kernels.'),
                    '_kernelconfbase'    : PathSpecs('kernel_nm_base', WhenMissing.IGNORE, CheckFile.MAKE_DIR, 'INNER/KERNELS/kernel_conf', 'Base path for per-molecule kernels.'),
                    '_goodcoeffilebase'  : PathSpecs('coeffs_base', WhenMissing.IGNORE, CheckFile.MAKE_DIR, 'INNER/coeff/mol', 'Base path for cleaned/reordered coefficients.'),
                    '_goodmetricfilebase': PathSpecs('metric_matrix_base', WhenMissing.IGNORE, CheckFile.MAKE_DIR, 'INNER/metric/mol', 'Base path for processed metric matrices.'),
                    '_baselinedwbase'    : PathSpecs('baselined_weights_base', WhenMissing.IGNORE, CheckFile.MAKE_DIR, 'INNER/BASELINED_PROJECTIONS/projections_conf', 'Base path for projected coefficients.'),
                    'spherical_averages' : PathSpecs('averages', WhenMissing.IGNORE, CheckFile.MAKE_DIR, 'INNER/AVERAGES.mts', 'File to store spherical averages.'),
                    'train_test_sets'    : PathSpecs('training_selection', WhenMissing.IGNORE, CheckFile.MAKE_DIR, 'INNER/SELECTIONS/training_selection.csv', 'CSV file to store train/test molecule indices.'),
                    'coef_norms'         : PathSpecs('coef_norms', WhenMissing.IGNORE, CheckFile.MAKE_DIR, 'INNER/norms.npy', 'Path for coefficient norms wrt the metric matrices.'),
                    '_targetvecfilebase' : PathSpecs('target_vector_base', WhenMissing.IGNORE, CheckFile.MAKE_DIR, 'INNER/target_vec', 'Base path for target-vector outputs.'),
                    '_grammatfilebase'   : PathSpecs('gram_matrix_base', WhenMissing.IGNORE, CheckFile.MAKE_DIR, 'INNER/gram_mat', 'Base path for Gram-matrix outputs.'),
                    '_mltermsfilebase'   : PathSpecs('ml_terms_base', WhenMissing.IGNORE, CheckFile.MAKE_DIR, 'INNER/ml_terms', 'Base path for PITC marginal-likelihood accumulator outputs.'),
                    },
                }

    def postprocess_options(parsed):
        """Convert raw parsed options to their final runtime representation.

        Args:
            parsed (dict[str, dict[str, object]]): Parsed values grouped by section name.

        Returns:
            types.SimpleNamespace: Post-processed options namespace.
        """
        o = dict(ChainMap(*[d for group, d in parsed.items() if group.startswith('options.')]))
        if o['use_charges']=='none':
            o['use_charges'] = None
        return SimpleNamespace(o)

    def postprocess_paths(parsed, o):
        """Derive path templates.

        Args:
            parsed (dict[str, dict[str, object]]): Parsed values grouped by section name.
            o (types.SimpleNamespace): Post-processed options namespace.

        Returns:
            types.SimpleNamespace: Namespace exposing all computed path templates.
        """
        p = dict(ChainMap(*[d for group, d in parsed.items() if group.startswith('paths.')]))

        paths = SimpleNamespace({key: val for key, val in p.items() if not key.startswith('_')})
        paths.clean_coefficients      = f'{p['_goodcoeffilebase']}_{{}}.npy'
        paths.metric_matrix           = f'{p['_goodmetricfilebase']}_{{}}.mts'
        paths.projection              = f'{p['_baselinedwbase']}{{}}.mts'

        paths.power_spectrum          = f'{p['_splitpsfilebase']}_{{}}.mts'
        paths.reference_environments  = f'{p['_refsselfilebase']}_{o.M}.csv'
        paths.reference_power_spectra = f'{p['_powerrefbase']}_{o.M}.mts'
        paths.kernel_mm               = f'{p['_kmmbase']}{o.M}.mts'
        paths.kernel_nm               = f'{p['_kernelconfbase']}{{}}.mts'

        paths.target_vec              = f'{p['_targetvecfilebase']}_M{o.M}_trainfrac{{train_frac}}.txt'
        paths.gram_mat                = f'{p['_grammatfilebase']}_M{o.M}_trainfrac{{train_frac}}.dat'
        paths.weights                 = f'{p['_weightsfilebase']}_M{o.M}_trainfrac{{train_frac}}_reg{o.reg}_jit{o.jit}.mts'
        paths.predictions             = f'{p['_predictfilebase']}_{{subset}}_M{o.M}_trainfrac{{train_frac}}_reg{o.reg}_jit{o.jit}.mts'
        paths.predicted_coeff         = f'{p['_outfilebase']}_tf{{train_frac}}_{{order}}_{{imol}}.dat'
        # PITC-only outputs
        # ml_terms is written next to the Gram matrix / target vector, so it follows their naming
        # (no reg suffix); sigma_f2 is written next to the weights, so it follows theirs.
        paths.ml_terms                = f'{p['_mltermsfilebase']}_M{o.M}_trainfrac{{train_frac}}.txt'
        paths.sigma_f2                = f'{p['_weightsfilebase']}_sigma_f2_M{o.M}_trainfrac{{train_frac}}_reg{o.reg}.txt'
        paths.cholesky_pitc           = f'{p['_weightsfilebase']}_cholesky_pitc_M{o.M}_trainfrac{{train_frac}}_reg{o.reg}.npy'
        paths.var_trace               = f'{p['_predictfilebase']}_vartrace_{{subset}}_M{o.M}_trainfrac{{train_frac}}_reg{o.reg}.csv'

        paths.extra_kernel_nm         = f'{p['_kernelexbase']}{{}}.mts'
        paths.extra_power_spectrum    = f'{p['_powerexbase']}_{{}}.mts'
        paths.extra_predicted_coeff   = f'{p['_outexfilebase']}_{{order}}_{{imol}}.dat'
        return paths

    config = Config(set_options() | set_paths())

    if print_help:
        config.print_help()
        sys.exit(0)

    parsed = config.parse(config_path=config_path)

    o = postprocess_options(parsed)
    p = postprocess_paths(parsed, o)
    return o, p
