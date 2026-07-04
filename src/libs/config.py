import sys
import os
import argparse
from types import SimpleNamespace
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
        print(f'Configuration file: {config_path}'+link)

        self.configuration = configparser.RawConfigParser()
        self.configuration.read(config_path)
        self.options = dict(self.configuration.items('options'))
        self.paths   = dict(self.configuration.items('paths'))

    def get_option(self, key, default, ttype):
        if key in self.options:
            return ttype(self.options[key])
        else:
            return default

    def floats(self, x):
        return np.unique(list(map(float, x.split(','))))

    def bool(self, x):
        x = x.lower()
        if x in ['1', 'true']:
            return True
        elif x in ['0', 'false']:
            return False
        msg = f'Wrong input for a Bool option: "{x}"'
        raise TypeError(msg)

    def choice(self, dtype, options, name=None, strict=True):
        def checker(x):
            x = dtype(x)
            if x not in options:
                if strict:
                    msg = f'Wrong input for `{name}`: `{x}` not in {options}'
                    raise RuntimeError(msg)
                else:
                    warn_short(f'`{x}` not in recommended options {options} for `{name}`')
            return x
        return checker


def read_config(argv, return_args=None):
    def set_variable_values():
        options = {
                'M'                  : ('m'                   , 100             , int         ),
                'seed'               : ('seed'                , 1               , int         ),
                'train'              : ('train_size'          , 1000            , int         ),
                'fracs'              : ('trainfrac'           , np.array([1.0]) , conf.floats ),
                'soap_sigma'         : ('soap_sigma'          , 0.3             , float       ),
                'soap_rcut'          : ('soap_rcut'           , 4.0             , float       ),
                'soap_ncut'          : ('soap_ncut'           , 8               , int         ),
                'soap_lcut'          : ('soap_lcut'           , 6               , int         ),
                'process_metric'     : ('process_metric'      , True            , conf.bool   ),
                'reg'                : ('regular'             , 1e-6            , float       ),
                'jit'                : ('jitter'              , 1e-10           , float       ),
                'use_charges'        : ('number_of_electrons' , 'none'          , conf.choice(str, ['none', 'charge', 'N'], name='number_of_electrons')),
                'ps_min_norm'        : ('ps_min_norm'         , 1e-20           , float       ),
                'ps_normalize'       : ('ps_normalize'        , True            , conf.bool   ),
                'basisname'          : ('basis'               , 'cc-pvqz-jkfit' , str         ),
                'coeff_order'        : ('coeff_order'         , 'pyscf'         , conf.choice(str, ['pyscf', 'gpr'], name='coeff_order',        strict=False)),
                'overlap_order'      : ('overlap_order'       , 'pyscf'         , conf.choice(str, ['pyscf', 'gpr'], name='overlap_order',      strict=False)),
                'output_coeff_order' : ('output_coeff_order'  , 'gpr'           , conf.choice(str, ['pyscf', 'gpr'], name='output_coeff_order', strict=False)),
                }

        recognized_options = [val[0] for val in options.values()]
        present_options = conf.options.keys()
        if unrecognized_options:=set(present_options).difference(recognized_options):
            msg = f'Unrecognized_options: {unrecognized_options}'
            raise RuntimeError(msg)
        return SimpleNamespace({dest: conf.get_option(*signature) for dest, signature in options.items()})

    def get_all_paths():

        paths = {
        'dataset'            : ('dataset'         , 'ERROR' ,  'ERROR_FILE', None),
        'xyz'                : ('xyz'             , 'ERROR' ,  'ERROR_DIR' , None),
        'input_metrics'      : ('metrics'         , 'ERROR' ,  'ERROR_DIR' , None),
        'input_coeffs'       : ('coeffs'          , 'ERROR' ,  'ERROR_DIR' , None),

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
        '_weightsfilebase'   : ('weights_base'    , None    ,  'MAKE_DIR'  , 'INNER/weights'),
        '_predictfilebase'   : ('predict_base'    , None    ,  'MAKE_DIR'  , 'INNER/prediction'),
        '_outfilebase'       : ('output_base'     , 'WARN'  ,  'MAKE_DIR'  , 'INNER/predicted/rho'),

        'xyzexfilename'      : ('ex_xyzfile'      , 'WARN'  ,  'ERROR_FILE', None),
        '_powerexbase'       : ('ex_ps_base'      , None    ,  'MAKE_DIR'  , 'INNER/extra/PS'),
        '_kernelexbase'      : ('ex_kernel_base'  , None    ,  'MAKE_DIR'  , 'INNER/extra/kernel'),
        '_outexfilebase'     : ('ex_output_base'  , 'WARN'  ,  'MAKE_DIR'  , 'INNER/extra/rho'),
        }

        recognized_paths = [val[0] for val in paths.values()]
        present_paths = conf.paths.keys()
        if unrecognized_paths:=set(present_paths).difference(recognized_paths):
            msg = f'Unrecognized_paths: {unrecognized_paths}'
            raise RuntimeError(msg)

        p = {}
        for dest, (option, when_missing_config, check_file, default) in paths.items():
            path = conf.paths.get(option)
            if path is None:
                if when_missing_config=='WARN':
                    warn_short(f'Missing recommended config path `{option}`')
                elif when_missing_config=='ERROR':
                    msg = f'Missing required config path `{option}`'
                    raise RuntimeError(msg)
                else:
                    path = default

            else:
                if check_file=='ERROR_FILE':
                    isfile = os.path.isfile(path)
                    if not isfile:
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

        return SimpleNamespace(p)

    args = parse_cli_args(argv[1:], return_args)
    conf = Config(config_path=args.config)
    o = set_variable_values()
    p = get_all_paths()

    if o.use_charges=='none':
        o.use_charges = None

    p.clean_coefficients      = f'{p._goodcoeffilebase}_{{}}.npy'
    p.metric_matrix           = f'{p._goodoverfilebase}_{{}}.mts'
    p.projection              = f'{p._baselinedwbase}{{}}.mts'

    p.power_spectrum          = f'{p._splitpsfilebase}_{{}}.mts'
    p.reference_environments  = f'{p._refsselfilebase}_{o.M}.csv'
    p.reference_power_spectra = f'{p._powerrefbase}_{o.M}.mts'
    p.kernel_mm               = f'{p._kmmbase}{o.M}.mts'
    p.kernel_nm               = f'{p._kernelconfbase}{{}}.mts'

    p.avec                    = f'{p._avecfilebase}_M{o.M}_trainfrac{{train_frac}}.txt'
    p.bmat                    = f'{p._bmatfilebase}_M{o.M}_trainfrac{{train_frac}}.dat'
    p.weights                 = f'{p._weightsfilebase}_M{o.M}_trainfrac{{train_frac}}_reg{o.reg}_jit{o.jit}.mts'
    p.predictions             = f'{p._predictfilebase}_{{subset}}_M{o.M}_trainfrac{{train_frac}}_reg{o.reg}_jit{o.jit}.mts'
    p.predicted_coeff         = f'{p._outfilebase}_tf{{train_frac}}_{{order}}_{{imol}}.dat'

    p.extra_kernel_nm         = f'{p._kernelexbase}{{}}.mts'
    p.extra_power_spectrum    = f'{p._powerexbase}_{{}}.mts'
    p.extra_predicted_coeff   = f'{p._outexfilebase}_{{order}}_{{imol}}.dat'

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
