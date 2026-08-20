"""Path and option specification assembly for project configuration."""

import sys
from types import SimpleNamespace
from collections import ChainMap
import logging
import numpy as np
from .config_parser import Config
from .config_utils import (CheckFile, WhenMissing, PathSpecs, OptSpecs, defaults, Floats, Bool, Choice,
                           RegressionModel, FloatOrFit, SAGPR, GPR_DTC, FIT_REG,
                           DEFAULT_DIR_GROUP, DEFAULT_DIR_KEY, DEFAULT_DIR_PLACEHOLDER)

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
                    'reg'                : OptSpecs('regularisation', 1e-6, FloatOrFit('regularisation'), f'Ridge regularization strength for regression (the sparse-GP noise scale eta). A positive float fixes it; `{FIT_REG}` makes regression.py choose it by maximizing the DTC marginal likelihood, which is only defined for regression_model = {GPR_DTC}.'),
                    'sigma_p2'           : OptSpecs('prior_scale', None, FloatOrFit('prior_scale'), f'Prior variance sigma_p^2 (main.pdf Sec. IIF): the single overall factor on the predictive variance. A positive float pins it; `{FIT_REG}` (the default) makes regression.py determine it by type-II maximum likelihood, Eq. 26. It does not enter the predictive mean, so changing it never changes a prediction.'),
                    'jit'                : OptSpecs('jitter', 1e-10, float, 'Diagonal regularization for regression, relative to each matrix mean diagonal (so one value suits K_MM, the metric and Sigma_M alike).'),
                    'regression_model'   : OptSpecs('regression_model', default=SAGPR, dtype=RegressionModel('regression_model'), help='Which model to fit. All three are sparse over the M reference environments. `sagpr` is the deterministic SA-GPR fit and produces no uncertainty. `gpr_DTC` and `gpr_PITC` are sparse Gaussian processes and unlock the predictive variance; gpr_DTC solves the very same linear system as sagpr and only adds the K**-Q** variance correction, while gpr_PITC also carries the per-molecule Nystrom residual D_i and so improves the mean at a higher assembly cost.'),
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

        Derived-data path defaults are written as `{default_dir}<relative>`, with no separator after
        the placeholder: the resolved prefix always ends with one, so an absolute `default_dir` works
        too. The parser substitutes the placeholder in configured values as well, so an override may
        reuse the prefix (`kmm_base = {default_dir}KMM_v2`) or escape it entirely (`kmm_base = ../shared/KMM`).

        Because every relocatable path has a default, `when_missing` follows the declared default:
        ERROR/WARN where it is None (missing entry means the feature is unavailable), IGNORE where a
        default exists.

        Returns:
            dict[str, dict[str, PathSpecs | OptSpecs]]: Specification of path groups and entries.
        """
        return {
                DEFAULT_DIR_GROUP: {
                    DEFAULT_DIR_KEY      : OptSpecs(DEFAULT_DIR_KEY, defaults.default_dir, str, f'Directory prefix substituted for every "{DEFAULT_DIR_PLACEHOLDER}" in the paths below, so that one line relocates the whole derived-data tree. A missing trailing separator is added; leave empty to write into the working directory.'),
                    },
                # Inputs are read-only and shared between runs, so they are never relocated.
                'paths.input': {
                    'dataset'            : PathSpecs('dataset', WhenMissing.ERROR, CheckFile.ERROR_FILE, None, 'CSV dataset with molecule IDs and optional charge columns.'),
                    'xyz'                : PathSpecs('xyz', WhenMissing.ERROR, CheckFile.ERROR_DIR, None, 'Template path to input XYZ files (uses {mol_name}).'),
                    'input_metrics'      : PathSpecs('metrics', WhenMissing.ERROR, CheckFile.ERROR_DIR, None, 'Template path to input metric matrices.'),
                    'input_coeffs'       : PathSpecs('coeffs', WhenMissing.ERROR, CheckFile.ERROR_DIR, None, 'Template path to input coefficient files.'),
                    },
                'paths.output': {
                    '_weightsfilebase'   : PathSpecs('weights_base', WhenMissing.IGNORE, CheckFile.MAKE_DIR, '{default_dir}weights', 'Base path for trained model weights.'),
                    '_predictfilebase'   : PathSpecs('predict_base', WhenMissing.IGNORE, CheckFile.MAKE_DIR, '{default_dir}prediction', 'Base path for predicted TensorMap coefficients.'),
                    '_outfilebase'       : PathSpecs('output_base', WhenMissing.IGNORE, CheckFile.MAKE_DIR, '{default_dir}predicted/rho', 'Base path for exported coefficient text files.'),
                    },
                'paths.extrapolation': {
                    'extra_xyzfilename'  : PathSpecs('xyz', WhenMissing.WARN, CheckFile.ERROR_FILE, None, 'XYZ file for extrapolation/out-of-sample molecules.'),
                    '_powerexbase'       : PathSpecs('power_spectra_base', WhenMissing.IGNORE, CheckFile.MAKE_DIR, '{default_dir}extra/PS', 'Base path for extrapolation/OOS power spectra.'),
                    '_kernelexbase'      : PathSpecs('kernel_base', WhenMissing.IGNORE, CheckFile.MAKE_DIR, '{default_dir}extra/kernel', 'Base path for extrapolation/OOS kernels.'),
                    '_outexfilebase'     : PathSpecs('output_base', WhenMissing.IGNORE, CheckFile.MAKE_DIR, '{default_dir}extra/rho', 'Base path for extrapolation/OOS exported coefficients.'),
                    },
                'paths.internal': {
                    'xyzfilename'        : PathSpecs('xyz', WhenMissing.IGNORE, CheckFile.MAKE_DIR, '{default_dir}dataset.xyz', 'Combined XYZ file written from all molecules.'),
                    '_splitpsfilebase'   : PathSpecs('power_spectra_split_base', WhenMissing.IGNORE, CheckFile.MAKE_DIR, '{default_dir}PS/PS', 'Base path for per-molecule power spectra.'),
                    '_refsselfilebase'   : PathSpecs('reference_selection_base', WhenMissing.IGNORE, CheckFile.MAKE_DIR, '{default_dir}SELECTIONS/refs_selection', 'Base path for selected reference environments.'),
                    '_powerrefbase'      : PathSpecs('power_spectra_reference_base', WhenMissing.IGNORE, CheckFile.MAKE_DIR, '{default_dir}PS', 'Base path for merged reference power spectra.'),
                    '_kmmbase'           : PathSpecs('kmm_base', WhenMissing.IGNORE, CheckFile.MAKE_DIR, '{default_dir}KMM', 'Base path for reference-reference kernels.'),
                    '_kernelconfbase'    : PathSpecs('kernel_nm_base', WhenMissing.IGNORE, CheckFile.MAKE_DIR, '{default_dir}KERNELS/kernel_conf', 'Base path for per-molecule kernels.'),
                    '_goodcoeffilebase'  : PathSpecs('coeffs_base', WhenMissing.IGNORE, CheckFile.MAKE_DIR, '{default_dir}coeff/mol', 'Base path for cleaned/reordered coefficients.'),
                    '_goodmetricfilebase': PathSpecs('metric_matrix_base', WhenMissing.IGNORE, CheckFile.MAKE_DIR, '{default_dir}metric/mol', 'Base path for processed metric matrices.'),
                    '_baselinedwbase'    : PathSpecs('baselined_weights_base', WhenMissing.IGNORE, CheckFile.MAKE_DIR, '{default_dir}BASELINED_PROJECTIONS/projections_conf', 'Base path for projected coefficients.'),
                    'spherical_averages' : PathSpecs('averages', WhenMissing.IGNORE, CheckFile.MAKE_DIR, '{default_dir}AVERAGES.mts', 'File to store spherical averages.'),
                    'train_test_sets'    : PathSpecs('training_selection', WhenMissing.IGNORE, CheckFile.MAKE_DIR, '{default_dir}SELECTIONS/training_selection.csv', 'CSV file to store train/test molecule indices.'),
                    'coef_norms'         : PathSpecs('coef_norms', WhenMissing.IGNORE, CheckFile.MAKE_DIR, '{default_dir}coef_norms.npy', 'Path for coefficient norms wrt the metric matrices.'),
                    '_targetvecfilebase' : PathSpecs('target_vector_base', WhenMissing.IGNORE, CheckFile.MAKE_DIR, '{default_dir}target_vec', 'Base path for target-vector outputs.'),
                    '_grammatfilebase'   : PathSpecs('gram_matrix_base', WhenMissing.IGNORE, CheckFile.MAKE_DIR, '{default_dir}gram_mat', 'Base path for Gram-matrix outputs.'),
                    '_mltermsfilebase'   : PathSpecs('ml_terms_base', WhenMissing.IGNORE, CheckFile.MAKE_DIR, '{default_dir}ml_terms', 'Base path for PITC marginal-likelihood accumulator outputs.'),
                    },
                }

    def postprocess_options(parsed):
        """Convert raw parsed options to their final runtime representation.

        Args:
            parsed (dict[str, dict[str, object]]): Parsed values grouped by section name.

        Returns:
            types.SimpleNamespace: Post-processed options namespace.

        Raises:
            RuntimeError: `regularisation = fit` was combined with a model whose assembly depends
                on eta, so the fit cannot be done in regression.py alone.
        """
        o = dict(ChainMap(*[d for group, d in parsed.items() if group.startswith('options.')]))
        if o['use_charges']=='none':
            o['use_charges'] = None
        o['is_gpr'] = o['regression_model']!=SAGPR
        o['fit_reg'] = o['reg'] is None
        if o['fit_reg'] and o['regression_model']!=GPR_DTC:
            # sagpr has no likelihood to maximize, and gpr_PITC's Lambda_i = D_i + eta*S_i^-1 puts
            # eta inside the assembly, so scanning it would mean re-running get_matrices.py per
            # trial value rather than one extra Cholesky in regression.py.
            msg = (f'`regularisation = {FIT_REG}` requires regression_model = {GPR_DTC}, '
                   f'got {o["regression_model"]}. Only DTC assembles a Gram matrix and target '
                   'vector that are independent of eta, which is what makes the fit cheap.')
            raise RuntimeError(msg)
        # Output file names carry eta. It is not known until regression.py has run when it is
        # fitted, so the tag stands in for it and the fitted value goes to p.fitted_reg. A fixed
        # eta formats exactly as before, so existing trees keep their file names.
        o['reg_tag'] = FIT_REG if o['fit_reg'] else o['reg']
        o['fit_sigma_p2'] = o['sigma_p2'] is None
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
        paths.weights                 = f'{p['_weightsfilebase']}_M{o.M}_trainfrac{{train_frac}}_reg{o.reg_tag}_jit{o.jit}.mts'
        paths.predictions             = f'{p['_predictfilebase']}_{{subset}}_M{o.M}_trainfrac{{train_frac}}_reg{o.reg_tag}_jit{o.jit}.mts'
        paths.predicted_coeff         = f'{p['_outfilebase']}_tf{{train_frac}}_{{order}}_{{imol}}.dat'
        # GP-only outputs (o.is_gpr)
        # ml_terms is written next to the Gram matrix / target vector, so it follows their naming
        # (no reg suffix); sigma_p2 is written next to the weights, so it follows theirs.
        # The Cholesky factor is of a model-dependent matrix -- Sigma_M for gpr_PITC, the SA-GPR
        # matrix A for gpr_DTC -- so the model is part of its name.
        paths.ml_terms                = f'{p['_mltermsfilebase']}_M{o.M}_trainfrac{{train_frac}}.txt'
        paths.sigma_p2                = f'{p['_weightsfilebase']}_sigma_p2_M{o.M}_trainfrac{{train_frac}}_reg{o.reg_tag}.txt'
        paths.cholesky                = f'{p['_weightsfilebase']}_cholesky_{o.regression_model}_M{o.M}_trainfrac{{train_frac}}_reg{o.reg_tag}.npy'
        # sigma_p^2 scales the variance and nothing else (main.pdf Eq. 21 does not contain it), so
        # it tags the variance table alone -- and only when pinned, leaving fitted runs' names as
        # they were. Weights and predictions are unaffected by it and keep their names either way.
        sp2_tag = '' if o.sigma_p2 is None else f'_sp2{o.sigma_p2}'
        paths.var_trace               = f'{p['_predictfilebase']}_vartrace_{{subset}}_M{o.M}_trainfrac{{train_frac}}_reg{o.reg_tag}{sp2_tag}.csv'
        # Written by regression.py only when eta is fitted; read back by whatever needs the number
        # itself rather than just a file name (variance_lib.dtc_lambda).
        paths.fitted_reg              = f'{p['_weightsfilebase']}_fitted_reg_M{o.M}_trainfrac{{train_frac}}.txt'

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
