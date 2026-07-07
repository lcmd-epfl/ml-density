"""Parse CLI arguments and a configuration file."""

import os
import argparse
from types import SimpleNamespace
from typing import NamedTuple
from enum import Enum
from collections.abc import Callable
from collections import ChainMap
from itertools import starmap
import configparser
import logging
import numpy as np

DEFAULT_CONFIG = 'config.txt'
DEFAULT_MPI = True
DEFAULT_LOGLEVEL = logging.DEBUG

logger = logging.getLogger('__main__')


class WhenMissing(Enum):
    """Policy for handling missing configuration entries."""
    IGNORE = 0
    WARN = 1
    ERROR = 2


class CheckFile(Enum):
    """Validation policy applied to configured paths."""
    ERROR_FILE = 1
    ERROR_DIR = 2
    MAKE_DIR = 3


class PathSpecs(NamedTuple):
    """Specification of one path option in a config section."""
    key: str
    when_missing: WhenMissing
    check_file: CheckFile
    default : str | None


class OptSpecs(NamedTuple):
    """Specification of one variable option in a config section."""
    key: str                             # Option key inside the section.
    default: object                      # Default value when the key is missing.
    dtype: Callable[[str], object]       # Callable converting raw string values.


class Config:
    """Read, validate, and parse configuration sections."""

    def __init__(self, config_path=DEFAULT_CONFIG):
        """Initialize a Config instance.

        Args:
            config_path (str | None): Path to the configuration file.

        Raises:
            RuntimeError: When the file is not found.
        """
        if not os.path.isfile(config_path):
            msg = f'Cannot open configuration file "{config_path}"'
            raise RuntimeError(msg)
        link = f' -> {os.readlink(config_path)}' if os.path.islink(config_path) else ''
        logger.info(f'Configuration file: {config_path}{link}')
        parser = configparser.RawConfigParser()
        parser.read(config_path)
        self.configuration = dict(parser.items())

    def get_option(self, group, specs):
        """Read and cast one option from a configuration section.

        Args:
            group (str): Configuration section name.
            specs (OptSpecs): Parsing specs.

        Returns:
            object: Parsed option value.
        """
        return specs.dtype(val) if (val := self.configuration[group].get(specs.key, None)) is not None else specs.default

    class Floats:
        """Parser converting comma-separated strings to unique float arrays."""
        def __call__(self, x):
            """Parse a comma-separated list of floats.

            Args:
                x (str): Raw input value.

            Returns:
                np.ndarray: Sorted unique float values.
            """
            return np.unique(list(map(float, x.split(','))))

    class Bool:
        """Parser converting common textual flags to booleans."""
        def __call__(self, x):
            """Parse a textual boolean option.

            Args:
                x (str): Raw input value.

            Returns:
                bool: Parsed and validated value.

            Raises:
                TypeError: x cannot be interpreted as boolean value.
            """
            x = x.lower()
            if x in {'1', 'true', 'on', 'yes'}:
                return True
            if x in {'0', 'false', 'off', 'no'}:
                return False
            msg = f'Wrong input for a Bool option: "{x}"'
            raise TypeError(msg)

    class Choice:
        """Parser wrapper enforcing membership in an allowed option set."""

        def __init__(self, dtype, options, name, *, strict=True):
            """Initialize a Choice instance.

            Args:
                dtype (Callable[[str], T]): Converter used to cast raw configuration values.
                options (list[T]): Accepted options (after casting).
                name (str): Option name (for logging/error messages).
                strict (bool): Whether invalid values should raise an error.
            """
            self.dtype = dtype
            self.options = options
            self.name = name
            self.strict = strict

        def __call__(self, x):
            """Cast and validate a value against the configured choices.

            Args:
                x (str): Raw input value.

            Returns:
                object: Validated value cast with self.dtype.

            Raises:
                RuntimeError: Value is not in accepted options and the validation is strict.
            """
            if (x := self.dtype(x)) not in self.options:
                if self.strict:
                    msg = f'Wrong input for `{self.name}`: `{x}` not in {self.options}'
                    raise RuntimeError(msg)
                logger.warning(f'`{x}` not in the recommended options {self.options} for `{self.name}`')
            return x

    def check_for_unrecognized_options(self, group, options):
        """Check for unrecognized options.

        Args:
            group (str): Configuration section name.
            options (dict[str, PathSpecs | OptSpecs]): Mapping from destination names to parsing specs.

        Raises:
            RuntimeError: There are unrecognized options in the config section.
        """
        recognized = [val.key for val in options.values()]
        present = self.configuration[group].keys()
        if unrecognized := set(present).difference(recognized):
            msg = f'Unrecognized entries in [{group}]: {unrecognized}'
            raise RuntimeError(msg)

    def get_option_group(self, group, options):
        """Read all options declared for one configuration section.

        Args:
            group (str): Configuration section name.
            options (dict[str, OptSpecs]): Mapping from destination names to parsing specs.

        Returns:
            dict[str, object]: Parsed options keyed by destination name.
        """
        self.check_for_unrecognized_options(group, options)
        return {dest: self.get_option(group, specs) for dest, specs in options.items()}

    def get_path_group(self, group, paths):
        """Read and validate all path entries for one section.

        Create empty directories if they are specified and missing.

        Args:
            group (str): Configuration section name.
            paths (dict[str, PathSpecs]): Mapping from destination names to parsing specs.

        Returns:
            dict[str, str]: Resolved path values keyed by destination name.

        Raises:
            RuntimeError: A required file does not exist.
            RuntimeError: A required directory does not exist.
        """
        self.check_for_unrecognized_options(group, paths)
        p = {}
        for dest, spec in paths.items():
            if (path := self.configuration[group].get(spec.key)) is None:
                if spec.when_missing==WhenMissing.WARN:
                    logger.warning(f'Missing recommended config path `{spec.key}`')
                elif spec.when_missing==WhenMissing.ERROR:
                    msg = f'Missing required config path `{spec.key}`'
                    raise RuntimeError(msg)
                else:
                    path = spec.default

            elif spec.check_file==CheckFile.ERROR_FILE:
                if not os.path.isfile(path):
                    msg = f'Missing file "{path}" ("{spec.key}")'
                    raise RuntimeError(msg)
            elif spec.check_file in {CheckFile.ERROR_DIR, CheckFile.MAKE_DIR}:
                fdir = os.path.dirname(path)
                if not os.path.isdir(fdir):
                    if spec.check_file==CheckFile.ERROR_DIR:
                        msg = f'Missing directory "{fdir}" ("{spec.key}")'
                        raise RuntimeError(msg)
                    logger.debug(f'Creating directory "{fdir}" ("{spec.key}")')
                    os.makedirs(fdir)
            p[dest] = path
        return p


def read_config(config_path=DEFAULT_CONFIG):
    """Read configuration options and derive runtime path templates.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        tuple[types.SimpleNamespace, types.SimpleNamespace]: Parsed options namespace and resolved paths namespace.
    """
    def set_variable_values():
        """Specify numeric/string options, load, parse, and validate.

        Returns:
            dict[str, object]: Flattened option dictionary ready for post-processing.
        """
        options = {
                'options.training': {
                    'M'                  : OptSpecs('reference_environments' , 100          , int         ),
                    'seed'               : OptSpecs('seed'                , 1               , int         ),
                    'train'              : OptSpecs('train_size'          , 1000            , int         ),
                    'fracs'              : OptSpecs('train_fractions'     , np.array([1.0]) , conf.Floats() ),
                    'reg'                : OptSpecs('regular'             , 1e-6            , float       ),
                    'jit'                : OptSpecs('jitter'              , 1e-10           , float       ),
                    },
                'options.soap': {
                    'soap_sigma'         : OptSpecs('soap_sigma'          , 0.3             , float       ),
                    'soap_rcut'          : OptSpecs('soap_rcut'           , 4.0             , float       ),
                    'soap_ncut'          : OptSpecs('soap_ncut'           , 8               , int         ),
                    'soap_lcut'          : OptSpecs('soap_lcut'           , 6               , int         ),
                    'ps_min_norm'        : OptSpecs('ps_min_norm'         , 1e-20           , float       ),
                    'ps_normalize'       : OptSpecs('ps_normalize'        , default=True    , dtype=conf.Bool() ),
                    },
                'options.rho': {
                    'process_metric'     : OptSpecs('process_metric'      , default=True    , dtype=conf.Bool() ),
                    'use_charges'        : OptSpecs('number_of_electrons' , 'none'          , conf.Choice(str, ['none', 'charge', 'N'], name='number_of_electrons')),
                    'basisname'          : OptSpecs('basis'               , 'cc-pvqz-jkfit' , str         ),
                    'coeff_order'        : OptSpecs('coeff_order'         , 'pyscf'         , conf.Choice(str, ['pyscf', 'gpr'], name='coeff_order',        strict=False)),
                    'overlap_order'      : OptSpecs('overlap_order'       , 'pyscf'         , conf.Choice(str, ['pyscf', 'gpr'], name='overlap_order',      strict=False)),
                    'output_coeff_order' : OptSpecs('output_coeff_order'  , 'gpr'           , conf.Choice(str, ['pyscf', 'gpr'], name='output_coeff_order', strict=False)),
                    },
                }
        return dict(ChainMap(*[*starmap(conf.get_option_group, options.items())]))

    def get_all_paths():
        """Specify path entries, load, parse, and validate.

        Returns:
            dict[str, str]: Flattened dictionary of configured and defaulted paths.
        """
        paths = {
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
        return dict(ChainMap(*[*starmap(conf.get_path_group, paths.items())]))

    def postprocess_options(o):
        """Convert raw parsed options to their final runtime representation.

        Args:
            o (dict[str, object]): Raw options dictionary from set_variable_values.

        Returns:
            types.SimpleNamespace: Post-processed options namespace.
        """
        if o['use_charges']=='none':
            o['use_charges'] = None
        return SimpleNamespace(o)

    def postprocess_paths(p, o):
        """Derive path templates.

        Args:
            p (dict[str, str]): Raw path dictionary from get_all_paths.
            o (types.SimpleNamespace): Post-processed options namespace.

        Returns:
            types.SimpleNamespace: Namespace exposing all computed path templates.
        """
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

    conf = Config(config_path=config_path)
    o = postprocess_options(set_variable_values())
    p = postprocess_paths(get_all_paths(), o)
    return o, p


def parse_cli_args(return_args=None):
    """Parse supported command-line arguments.

    Args:
        return_args (list[str] | None): Optional whitelist of extra arguments to expose.

    Returns:
        argparse.Namespace: Parsed CLI arguments namespace.
    """
    if return_args is None:
        return_args = []

    def add_argument(parser, *kargs, **kwargs):
        """Conditionally register an argument only when requested by return_args.

        Args:
            parser (argparse.ArgumentParser): Parser that may receive the argument.
            *kargs (object): Positional arguments forwarded to add_argument.
            **kwargs (object): Keyword arguments forwarded to add_argument.
        """
        if kwargs.get('dest') in return_args:
            parser.add_argument(*kargs, **kwargs)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help='path to the configuration file')
    parser.add_argument("--log", type=str, default=logging._levelToName[DEFAULT_LOGLEVEL], choices=logging._nameToLevel.keys(), help='logging level')
    add_argument(parser, "--training", dest='training', action='store_true', help='run prediction / compute error on the training set instead of the test set')
    add_argument(parser, "-b", "--b", dest="get_b_matrix", action='store_true', help='if True, get_matrices computes the "B matrix"; if False, the "A vector"')
    add_argument(parser, "--missing-only", dest="missing_only", action='store_true', help='dangerous: not recompute existing power spectra / kernels')

    mpi = parser.add_mutually_exclusive_group()
    mpi.add_argument("--mpi-dummy-argument", dest="mpi", default=DEFAULT_MPI, action='store_true', help=argparse.SUPPRESS)
    add_argument(mpi, "--mpi", dest="mpi", default=DEFAULT_MPI, action='store_true', help='set MPI usage flag')
    add_argument(mpi, "--no-mpi", dest="mpi", default=DEFAULT_MPI, action='store_false', help='set MPI usage flag')
    return parser.parse_args()


def get_settings(return_args=None):
    """Load CLI args and configuration options/paths for a script.

    Args:
        return_args (list[str] | None): Optional list of script-specific CLI flags to parse.

    Returns:
        tuple: Either (options, paths) or (args, options, paths) depending on return_args.
    """
    args = parse_cli_args(return_args)
    logger.setLevel(args.log)
    o, p = read_config(args.config)
    return (args, o, p) if return_args else (o, p)
