import sys
import os
from types import SimpleNamespace
import configparser
import numpy as np
from libs.functions import warn_short

DEFAULT_PATH = 'config.txt'


class Config:
    def __init__(self, config_path=DEFAULT_PATH):
        if config_path is None:
            config_path = DEFAULT_PATH
        if not os.path.isfile(config_path):
            raise RuntimeError(f'Cannot open configuration file "{config_path}"')
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
        raise TypeError(f'Wrong input for the Bool option: "{x}"')

    def choice(self, dtype, options, strict=True):
        def checker(x):
            x = dtype(x)
            if x not in options:
                if strict:
                    raise RuntimeError(f'Wrong input: {x} not in {options}')
                else:
                    warn_short(f'{x} not in recommended options {options}')
            return x
        return checker


def get_config_path(argv):
    path = None
    paths = ([x for x in argv[1:] if x.startswith('--config=')])
    if paths:
        path = paths[-1][len('--config='):]
    return path


def read_config(argv):
    def set_variable_values():
        o = SimpleNamespace()
        o.M                  = conf.get_option('m'                   , 100             , int         )
        o.seed               = conf.get_option('seed'                , 1               , int         )
        o.train              = conf.get_option('train_size'          , 1000            , int         )
        o.fracs              = conf.get_option('trainfrac'           , np.array([1.0]) , conf.floats )
        o.soap_sigma         = conf.get_option('soap_sigma'          , 0.3             , float       )
        o.soap_rcut          = conf.get_option('soap_rcut'           , 4.0             , float       )
        o.soap_ncut          = conf.get_option('soap_ncut'           , 8               , int         )
        o.soap_lcut          = conf.get_option('soap_lcut '          , 6               , int         )
        o.process_metric     = conf.get_option('process_metric'      , True            , conf.bool   )
        o.reg                = conf.get_option('regular'             , 1e-6            , float       )
        o.jit                = conf.get_option('jitter'              , 1e-10           , float       )
        o.use_charges        = conf.get_option('number_of_electrons' , 'none'          , conf.choice(str, ['none', 'charge', 'N']))
        o.ps_min_norm        = conf.get_option('ps_min_norm'         , 1e-20           , float       )
        o.ps_normalize       = conf.get_option('ps_normalize'        , True            , conf.bool   )
        o.basisname          = conf.get_option('basis'               , 'cc-pvqz-jkfit' , str         )
        o.coeff_order        = conf.get_option('coeff_order'         , 'pyscf'         , conf.choice(str, ['pyscf', 'gpr'], strict=False)  )
        o.overlap_order      = conf.get_option('overlap_order'       , 'pyscf'         , conf.choice(str, ['pyscf', 'gpr'], strict=False)  )
        o.output_coeff_order = conf.get_option('output_coeff_order'  , 'gpr'           , conf.choice(str, ['pyscf', 'gpr'], strict=False)  )
        return o

    def get_all_paths():
        p = SimpleNamespace()

        p.dataset = conf.paths.get('dataset')
        p.xyz = conf.paths.get('xyz')
        p.input_metrics = conf.paths.get('metrics')
        p.input_coeffs  = conf.paths.get('coeffs')

        p.xyzfilename       = conf.paths.get('xyzfile')
        p._splitpsfilebase  = conf.paths.get('ps_split_base')
        p._refsselfilebase  = conf.paths.get('refs_sel_base')
        p._powerrefbase     = conf.paths.get('ps_ref_base')

        p._kmmbase          = conf.paths.get('kmm_base')
        p.kernelconfbase   = conf.paths.get('kernel_conf_base')

        p._goodcoeffilebase = conf.paths.get('goodcoef_base')
        p.goodoverfilebase = conf.paths.get('goodover_base')
        p.baselinedwbase   = conf.paths.get('baselined_w_base')

        p.spherical_averages = conf.paths.get('averages_file')
        p.train_test_sets    = conf.paths.get('trainingselfile')

        p._avecfilebase     = conf.paths.get('avec_base')
        p._bmatfilebase     = conf.paths.get('bmat_base')
        p._weightsfilebase  = conf.paths.get('weights_base')
        p._predictfilebase  = conf.paths.get('predict_base')
        p._outfilebase      = conf.paths.get('output_base')

        p.xyzexfilename    = conf.paths.get('ex_xyzfile')
        p._powerexbase      = conf.paths.get('ex_ps_base')
        p._kernelexbase     = conf.paths.get('ex_kernel_base')
        p._outexfilebase    = conf.paths.get('ex_output_base')
        return p

    path = get_config_path(argv)
    conf = Config(config_path=path)
    check_paths(conf)
    o = set_variable_values()
    p = get_all_paths()

    if o.use_charges=='none':
        o.use_charges = None

    p.clean_coefficients      = f'{p._goodcoeffilebase}_{{}}.npy'
    p.metric_matrix           = f'{p.goodoverfilebase}{{}}.mts'
    p.projection              = f'{p.baselinedwbase}{{}}.mts'

    p.power_spectrum          = f'{p._splitpsfilebase}_{{}}.mts'
    p.reference_environments  = f'{p._refsselfilebase}_{o.M}.csv'
    p.reference_power_spectra = f'{p._powerrefbase}_{o.M}.mts'
    p.kernel_mm               = f'{p._kmmbase}{o.M}.mts'
    p.kernel_nm               = f'{p.kernelconfbase}{{}}.mts'

    p.avec                    = f'{p._avecfilebase}_M{o.M}_trainfrac{{train_frac}}.txt'
    p.bmat                    = f'{p._bmatfilebase}_M{o.M}_trainfrac{{train_frac}}.dat'
    p.weights                 = f'{p._weightsfilebase}_M{o.M}_trainfrac{{train_frac}}_reg{o.reg}_jit{o.jit}.mts'
    p.predictions             = f'{p._predictfilebase}_{{subset}}_M{o.M}_trainfrac{{train_frac}}_reg{o.reg}_jit{o.jit}.mts'
    p.predicted_coeff         = f'{p._outfilebase}_tf{{train_frac}}_{{order}}_{{imol}}.dat'

    p.extra_kernel_nm         = f'{p._kernelexbase}{{}}.mts'
    p.extra_power_spectrum    = f'{p._powerexbase}_{{}}.mts'
    p.extra_predicted_coeff   = f'{p._outexfilebase}_{{order}}_{{imol}}.dat'

    return o, p


def check_paths(conf):
    paths0 = [
      'dataset',
      'ex_xyzfile',
      'averages_file',
      ]

    paths1 = [
      'avec_base',
      'baselined_w_base',
      'bmat_base',
      'ex_kernel_base',
      'ex_output_base',
      'goodcoef_base',
      'goodover_base',
      'kernel_conf_base',
      'kmm_base',
      'output_base',
      'predict_base',
      'ps_ref_base',
      'ps_split_base',
      'refs_sel_base',
      'weights_base',
      'trainingselfile',
      'ex_ps_base',
      ]

    for key in paths0:
        if key in conf.paths:
            path = conf.paths[key]
            isfile = os.path.isfile(path)
            if not isfile:
                print(f'Cannot find file "{path}" ("{key}")')
        else:
            print(f'Cannot find option "{key}"')

    dirs = []
    for key in paths1:
        if key in conf.paths:
            path = conf.paths[key]
            path = os.path.dirname(path)
            isdir = os.path.isdir(path)
            if not isdir:
                #print(f'Cannot find directory "{path}" ("{key}")')
                dirs.append(path)
        else:
            print(f'Cannot find option "{key}"')
    print()

    for d in sorted(set(dirs)):
        print(f'Creating directory {d}')
        os.makedirs(d)
