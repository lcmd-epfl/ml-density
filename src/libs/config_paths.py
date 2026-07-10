"""Path and option specification assembly for project configuration."""

from types import SimpleNamespace
from collections import ChainMap
import logging
import numpy as np
from .config_parser import Config
from .config_utils import CheckFile, WhenMissing, PathSpecs, OptSpecs, defaults, Floats, Bool, Choice

logger = logging.getLogger('__main__')


def read_config(config_path=defaults.config):
    """Read configuration options and derive runtime path templates.

    Args:
        config_path (str): Path to the configuration file.

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
                    'M'                  : OptSpecs('reference_environments' , 100          , int         ),
                    'seed'               : OptSpecs('seed'                , 1               , int         ),
                    'train'              : OptSpecs('train_size'          , 1000            , int         ),
                    'fracs'              : OptSpecs('train_fractions'     , np.array([1.0]) , Floats()    ),
                    'reg'                : OptSpecs('regular'             , 1e-6            , float       ),
                    'jit'                : OptSpecs('jitter'              , 1e-10           , float       ),
                    },
                'options.soap': {
                    'soap_sigma'         : OptSpecs('soap_sigma'          , 0.3             , float       ),
                    'soap_rcut'          : OptSpecs('soap_rcut'           , 4.0             , float       ),
                    'soap_ncut'          : OptSpecs('soap_ncut'           , 8               , int         ),
                    'soap_lcut'          : OptSpecs('soap_lcut'           , 6               , int         ),
                    'ps_min_norm'        : OptSpecs('ps_min_norm'         , 1e-20           , float       ),
                    'ps_normalize'       : OptSpecs('ps_normalize'        , default=True    , dtype=Bool() ),
                    },
                'options.rho': {
                    'process_metric'     : OptSpecs('process_metric'      , default=True    , dtype=Bool() ),
                    'use_charges'        : OptSpecs('number_of_electrons' , 'none'          , Choice(str, ['none', 'charge', 'N'], name='number_of_electrons')),
                    'basisname'          : OptSpecs('basis'               , 'cc-pvqz-jkfit' , str         ),
                    'coeff_order'        : OptSpecs('coeff_order'         , 'pyscf'         , Choice(str, ['pyscf', 'gpr'], name='coeff_order',        strict=False)),
                    'overlap_order'      : OptSpecs('overlap_order'       , 'pyscf'         , Choice(str, ['pyscf', 'gpr'], name='overlap_order',      strict=False)),
                    'output_coeff_order' : OptSpecs('output_coeff_order'  , 'gpr'           , Choice(str, ['pyscf', 'gpr'], name='output_coeff_order', strict=False)),
                    },
                }

    def set_paths():
        """Build path specifications by configuration section.

        Returns:
            dict[str, dict[str, PathSpecs]]: Specification of path groups and entries.
        """
        return {
                'paths.input': {
                    'dataset'            : PathSpecs('dataset'         , WhenMissing.ERROR ,  CheckFile.ERROR_FILE, None),
                    'xyz'                : PathSpecs('xyz'             , WhenMissing.ERROR ,  CheckFile.ERROR_DIR , None),
                    'input_metrics'      : PathSpecs('metrics'         , WhenMissing.ERROR ,  CheckFile.ERROR_DIR , None),
                    'input_coeffs'       : PathSpecs('coeffs'          , WhenMissing.ERROR ,  CheckFile.ERROR_DIR , None),
                    },
                'paths.internal': {
                    'xyzfilename'        : PathSpecs('xyzfile'         , WhenMissing.IGNORE,  CheckFile.MAKE_DIR  , 'INNER/dataset.xyz'),
                    '_splitpsfilebase'   : PathSpecs('ps_split_base'   , WhenMissing.IGNORE,  CheckFile.MAKE_DIR  , 'INNER/PS/PS'),
                    '_refsselfilebase'   : PathSpecs('refs_sel_base'   , WhenMissing.IGNORE,  CheckFile.MAKE_DIR  , 'INNER/SELECTIONS/refs_selection'),
                    '_powerrefbase'      : PathSpecs('ps_ref_base'     , WhenMissing.IGNORE,  CheckFile.MAKE_DIR  , 'INNER/PS'),
                    '_kmmbase'           : PathSpecs('kmm_base'        , WhenMissing.IGNORE,  CheckFile.MAKE_DIR  , 'INNER/KMM'),
                    '_kernelconfbase'    : PathSpecs('kernel_conf_base', WhenMissing.IGNORE,  CheckFile.MAKE_DIR  , 'INNER/KERNELS/kernel_conf'),
                    '_goodcoeffilebase'  : PathSpecs('goodcoef_base'   , WhenMissing.IGNORE,  CheckFile.MAKE_DIR  , 'INNER/coeff/mol'),
                    '_goodoverfilebase'  : PathSpecs('goodover_base'   , WhenMissing.IGNORE,  CheckFile.MAKE_DIR  , 'INNER/metric/mol'),
                    '_baselinedwbase'    : PathSpecs('baselined_w_base', WhenMissing.IGNORE,  CheckFile.MAKE_DIR  , 'INNER/BASELINED_PROJECTIONS/projections_conf'),
                    'spherical_averages' : PathSpecs('averages_file'   , WhenMissing.IGNORE,  CheckFile.MAKE_DIR  , 'INNER/AVERAGES.mts'),
                    'train_test_sets'    : PathSpecs('trainingselfile' , WhenMissing.IGNORE,  CheckFile.MAKE_DIR  , 'INNER/SELECTIONS/training_selection.csv'),
                    '_avecfilebase'      : PathSpecs('avec_base'       , WhenMissing.IGNORE,  CheckFile.MAKE_DIR  , 'INNER/Avec'),
                    '_bmatfilebase'      : PathSpecs('bmat_base'       , WhenMissing.IGNORE,  CheckFile.MAKE_DIR  , 'INNER/Bmat'),
                    },
                'paths.output': {
                    '_weightsfilebase'   : PathSpecs('weights_base'    , WhenMissing.IGNORE,  CheckFile.MAKE_DIR  , 'INNER/weights'),
                    '_predictfilebase'   : PathSpecs('predict_base'    , WhenMissing.IGNORE,  CheckFile.MAKE_DIR  , 'INNER/prediction'),
                    '_outfilebase'       : PathSpecs('output_base'     , WhenMissing.WARN  ,  CheckFile.MAKE_DIR  , 'INNER/predicted/rho'),
                    },
                'paths.extrapolation': {
                    'xyzexfilename'      : PathSpecs('xyzfile'         , WhenMissing.WARN  ,  CheckFile.ERROR_FILE, None),
                    '_powerexbase'       : PathSpecs('ps_base'         , WhenMissing.IGNORE,  CheckFile.MAKE_DIR  , 'INNER/extra/PS'),
                    '_kernelexbase'      : PathSpecs('kernel_base'     , WhenMissing.IGNORE,  CheckFile.MAKE_DIR  , 'INNER/extra/kernel'),
                    '_outexfilebase'     : PathSpecs('output_base'     , WhenMissing.WARN  ,  CheckFile.MAKE_DIR  , 'INNER/extra/rho'),
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
        paths.metric_matrix           = f'{p['_goodoverfilebase']}_{{}}.mts'
        paths.projection              = f'{p['_baselinedwbase']}{{}}.mts'

        paths.power_spectrum          = f'{p['_splitpsfilebase']}_{{}}.mts'
        paths.reference_environments  = f'{p['_refsselfilebase']}_{o.M}.csv'
        paths.reference_power_spectra = f'{p['_powerrefbase']}_{o.M}.mts'
        paths.kernel_mm               = f'{p['_kmmbase']}{o.M}.mts'
        paths.kernel_nm               = f'{p['_kernelconfbase']}{{}}.mts'

        paths.avec                    = f'{p['_avecfilebase']}_M{o.M}_trainfrac{{train_frac}}.txt'
        paths.bmat                    = f'{p['_bmatfilebase']}_M{o.M}_trainfrac{{train_frac}}.dat'
        paths.weights                 = f'{p['_weightsfilebase']}_M{o.M}_trainfrac{{train_frac}}_reg{o.reg}_jit{o.jit}.mts'
        paths.predictions             = f'{p['_predictfilebase']}_{{subset}}_M{o.M}_trainfrac{{train_frac}}_reg{o.reg}_jit{o.jit}.mts'
        paths.predicted_coeff         = f'{p['_outfilebase']}_tf{{train_frac}}_{{order}}_{{imol}}.dat'

        paths.extra_kernel_nm         = f'{p['_kernelexbase']}{{}}.mts'
        paths.extra_power_spectrum    = f'{p['_powerexbase']}_{{}}.mts'
        paths.extra_predicted_coeff   = f'{p['_outexfilebase']}_{{order}}_{{imol}}.dat'
        return paths

    config = Config(config_path=config_path)

    for group, entries in (set_options() | set_paths()).items():
        config.add_group(group)
        for dest, spec in entries.items():
            config.add_entry(group, dest, spec)

    parsed = config.parse()

    o = postprocess_options(parsed)
    p = postprocess_paths(parsed, o)
    return o, p
