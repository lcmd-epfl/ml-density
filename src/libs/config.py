import sys
import os
import argparse
from types import SimpleNamespace
from collections import ChainMap
from itertools import starmap
import configparser
import numpy as np
from libs.functions import warn_short

DEFAULT_PATH = 'config.txt'
DEFAULT_MPI = True


class Config:
    def __init__(self, config_path=DEFAULT_PATH):
        if config_path is None:
            config_path = DEFAULT_PATH
        if not os.path.isfile(config_path):
            msg = f'Cannot open configuration file "{config_path}"'
            raise RuntimeError(msg)
        link = f' -> {os.readlink(config_path)}' if os.path.islink(config_path) else ''
        print(f'================ {sys.argv[0]} ================')
        print(f'Configuration file: {config_path}{link}')
        parser = configparser.RawConfigParser()
        parser.read(config_path)
        self.configuration = dict(parser.items())

    def get_option(self, group, key, default, dtype):
        if key in self.configuration[group]:
            return dtype(self.configuration[group][key])
        else:
            return default

    class Floats:
        def __call__(self, x):
            return np.unique(list(map(float, x.split(','))))

    class Bool:
        def __call__(self, x):
            x = x.lower()
            if x in ['1', 'true', 'on', 'yes']:
                return True
            elif x in ['0', 'false', 'off', 'no']:
                return False
            msg = f'Wrong input for a Bool option: "{x}"'
            raise TypeError(msg)

    class Choice:
        def __init__(self, dtype, options, name, strict=True):
            self.dtype = dtype
            self.options = options
            self.name = name
            self.strict = strict

        def __call__(self, x):
            if (x := self.dtype(x)) not in self.options:
                if self.strict:
                    msg = f'Wrong input for `{self.name}`: `{x}` not in {self.options}'
                    raise RuntimeError(msg)
                else:
                    warn_short(f'`{x}` not in the recommended options {self.options} for `{self.name}`')
            return x

    def check_for_unrecognized_options(self, group, options):
        recognized = [val[0] for val in options.values()]
        present = self.configuration[group].keys()
        if unrecognized := set(present).difference(recognized):
            msg = f'Unrecognized entries in [{group}]: {unrecognized}'
            raise RuntimeError(msg)

    def get_option_group(self, group, options):
        self.check_for_unrecognized_options(group, options)
        return {dest: self.get_option(group, key, default, dtype) for dest, (key, default, dtype) in options.items()}

    def get_path_group(self, group, paths):
        self.check_for_unrecognized_options(group, paths)
        p = {}
        for dest, (option, when_missing_config, check_file, default) in paths.items():
            if (path := self.configuration[group].get(option)) is None:
                if when_missing_config=='WARN':
                    warn_short(f'Missing recommended config path `{option}`')
                elif when_missing_config=='ERROR':
                    msg = f'Missing required config path `{option}`'
                    raise RuntimeError(msg)
                else:
                    path = default

            else:
                if check_file=='ERROR_FILE':
                    if not os.path.isfile(path):
                        msg = f'Missing file "{path}" ("{option}")'
                        raise RuntimeError(msg)
                elif check_file in ['ERROR_DIR', 'MAKE_DIR'] :
                    fdir = os.path.dirname(path)
                    if not os.path.isdir(fdir):
                        if check_file=='ERROR_DIR':
                            msg = f'Missing directory "{fdir}" ("{option}")'
                            raise RuntimeError(msg)
                        else:
                            warn_short(f'Creating directory "{fdir}" ("option")')
                            os.makedirs(fdir)
            p[dest] = path
        return p


def read_config(argv, return_args=None):

    def set_variable_values():
        options = {
                'options.training': {
                    'M'                  : ('reference_environments' , 100          , int         ),
                    'seed'               : ('seed'                , 1               , int         ),
                    'train'              : ('train_size'          , 1000            , int         ),
                    'fracs'              : ('train_fractions'     , np.array([1.0]) , conf.Floats() ),
                    'reg'                : ('regular'             , 1e-6            , float       ),
                    'jit'                : ('jitter'              , 1e-10           , float       ),
                    },
                'options.soap': {
                    'soap_sigma'         : ('soap_sigma'          , 0.3             , float       ),
                    'soap_rcut'          : ('soap_rcut'           , 4.0             , float       ),
                    'soap_ncut'          : ('soap_ncut'           , 8               , int         ),
                    'soap_lcut'          : ('soap_lcut'           , 6               , int         ),
                    'ps_min_norm'        : ('ps_min_norm'         , 1e-20           , float       ),
                    'ps_normalize'       : ('ps_normalize'        , True            , conf.Bool() ),
                    },
                'options.rho': {
                    'process_metric'     : ('process_metric'      , True            , conf.Bool() ),
                    'use_charges'        : ('number_of_electrons' , 'none'          , conf.Choice(str, ['none', 'charge', 'N'], name='number_of_electrons')),
                    'basisname'          : ('basis'               , 'cc-pvqz-jkfit' , str         ),
                    'coeff_order'        : ('coeff_order'         , 'pyscf'         , conf.Choice(str, ['pyscf', 'gpr'], name='coeff_order',        strict=False)),
                    'overlap_order'      : ('overlap_order'       , 'pyscf'         , conf.Choice(str, ['pyscf', 'gpr'], name='overlap_order',      strict=False)),
                    'output_coeff_order' : ('output_coeff_order'  , 'gpr'           , conf.Choice(str, ['pyscf', 'gpr'], name='output_coeff_order', strict=False)),
                    },
                }
        return dict(ChainMap(*[*starmap(conf.get_option_group, options.items())]))

    def get_all_paths():
        paths = {
                'paths.input': {
                    'dataset'            : ('dataset'         , 'ERROR' ,  'ERROR_FILE', None),
                    'xyz'                : ('xyz'             , 'ERROR' ,  'ERROR_DIR' , None),
                    'input_metrics'      : ('metrics'         , 'ERROR' ,  'ERROR_DIR' , None),
                    'input_coeffs'       : ('coeffs'          , 'ERROR' ,  'ERROR_DIR' , None),
                    },
                'paths.internal': {
                    'xyzfilename'        : ('xyzfile'         , None    ,  'MAKE_DIR'  , 'INNER/dataset.xyz'),
                    '_splitpsfilebase'   : ('ps_split_base'   , None    ,  'MAKE_DIR'  , 'INNER/PS/PS'),
                    '_refsselfilebase'   : ('refs_sel_base'   , None    ,  'MAKE_DIR'  , 'INNER/SELECTIONS/refs_selection'),
                    '_powerrefbase'      : ('ps_ref_base'     , None    ,  'MAKE_DIR'  , 'INNER/PS'),
                    '_kmmbase'           : ('kmm_base'        , None    ,  'MAKE_DIR'  , 'INNER/KMM'),
                    '_kernelconfbase'    : ('kernel_conf_base', None    ,  'MAKE_DIR'  , 'INNER/KERNELS/kernel_conf'),
                    '_goodcoeffilebase'  : ('goodcoef_base'   , None    ,  'MAKE_DIR'  , 'INNER/coeff/mol'),
                    '_goodoverfilebase'  : ('goodover_base'   , None    ,  'MAKE_DIR'  , 'INNER/metric/mol'),
                    '_baselinedwbase'    : ('baselined_w_base', None    ,  'MAKE_DIR'  , 'INNER/BASELINED_PROJECTIONS/projections_conf'),
                    'spherical_averages' : ('averages_file'   , None    ,  'MAKE_DIR'  , 'INNER/AVERAGES.mts'),
                    'train_test_sets'    : ('trainingselfile' , None    ,  'MAKE_DIR'  , 'INNER/SELECTIONS/training_selection.csv'),
                    '_avecfilebase'      : ('avec_base'       , None    ,  'MAKE_DIR'  , 'INNER/Avec'),
                    '_bmatfilebase'      : ('bmat_base'       , None    ,  'MAKE_DIR'  , 'INNER/Bmat'),
                    },
                'paths.output': {
                    '_weightsfilebase'   : ('weights_base'    , None    ,  'MAKE_DIR'  , 'INNER/weights'),
                    '_predictfilebase'   : ('predict_base'    , None    ,  'MAKE_DIR'  , 'INNER/prediction'),
                    '_outfilebase'       : ('output_base'     , 'WARN'  ,  'MAKE_DIR'  , 'INNER/predicted/rho'),
                    },
                'paths.extrapolation': {
                    'xyzexfilename'      : ('xyzfile'         , 'WARN'  ,  'ERROR_FILE', None),
                    '_powerexbase'       : ('ps_base'         , None    ,  'MAKE_DIR'  , 'INNER/extra/PS'),
                    '_kernelexbase'      : ('kernel_base'     , None    ,  'MAKE_DIR'  , 'INNER/extra/kernel'),
                    '_outexfilebase'     : ('output_base'     , 'WARN'  ,  'MAKE_DIR'  , 'INNER/extra/rho'),
                    },
                }
        return dict(ChainMap(*[*starmap(conf.get_path_group, paths.items())]))

    def postprocess_options(o):
        if o['use_charges']=='none':
            o['use_charges'] = None
        return SimpleNamespace(o)

    def postprocess_paths(p, o):
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

    args = parse_cli_args(argv[1:], return_args)
    conf = Config(config_path=args.config)

    o = postprocess_options(set_variable_values())
    p = postprocess_paths(get_all_paths(), o)

    if return_args is None:
        return o, p
    else:
        return args, o, p


def parse_cli_args(argv, return_args=None):
    if return_args is None:
        return_args = []

    def add_argument(parser, *kargs, **kwargs):
        if kwargs.get('dest') in return_args:
            parser.add_argument(*kargs, **kwargs)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, default=DEFAULT_PATH, help='path to the configuration file')
    add_argument(parser, "--training", dest='training', action='store_true', help='run prediction / compute error on the training set instead of the test set')
    add_argument(parser, "-b", "--b", dest="get_b_matrix", action='store_true', help='if True, get_matrices computes the "B matrix"; if False, the "A vector"')
    add_argument(parser, "--missing-only", dest="missing_only", action='store_true', help='dangerous: not recompute existing power spectra / kernels')

    mpi = parser.add_mutually_exclusive_group()
    mpi.add_argument("--mpi-dummy-argument", dest="mpi", default=DEFAULT_MPI, action='store_true', help=argparse.SUPPRESS)
    add_argument(mpi, "--mpi", dest="mpi", default=DEFAULT_MPI, action='store_true', help='set MPI usage flag')
    add_argument(mpi, "--no-mpi", dest="mpi", default=DEFAULT_MPI, action='store_false', help='set MPI usage flag')
    args = parser.parse_args(argv)
    return args
