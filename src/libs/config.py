import sys
import os
from types import SimpleNamespace
import configparser
import numpy as np

DEFAULT_PATH = 'config.txt'


class Config:
    def __init__(self, config_path=DEFAULT_PATH):
        if config_path is None:
            config_path = DEFAULT_PATH
        if not os.path.isfile(config_path):
            print(f'Cannot open configuration file "{config_path}"')
            exit(1)
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
        print(f'Wrong input for the Bool option: "{x}"')
        exit(0)


def get_config_path(argv):
    path = None
    paths = ([x for x in argv[1:] if x.startswith('--config=')])
    if paths:
        path = paths[-1][len('--config='):]
    return path


def read_config(argv):
    def set_variable_values():
        o = SimpleNamespace()
        o.M             = conf.get_option('m'            , 100             , int         )
        o.seed          = conf.get_option('seed'         , 1               , int         )
        o.train         = conf.get_option('train_size'   , 1000            , int         )
        o.fracs         = conf.get_option('trainfrac'    , np.array([1.0]) , conf.floats )
        o.soap_sigma    = conf.get_option('soap_sigma'   , 0.3             , float       )
        o.soap_rcut     = conf.get_option('soap_rcut'    , 4.0             , float       )
        o.soap_ncut     = conf.get_option('soap_ncut'    , 8               , int         )
        o.soap_lcut     = conf.get_option('soap_lcut '   , 6               , int         )
        o.reorder_ao    = conf.get_option('reorder_ao'   , 0               , int         )
        o.copy_metric   = conf.get_option('copy_metric'  , 1               , int         )
        o.reg           = conf.get_option('regular'      , 1e-6            , float       )
        o.jit           = conf.get_option('jitter'       , 1e-10           , float       )
        o.use_charges   = conf.get_option('charges'      , 0               , int         )
        o.ps_min_norm   = conf.get_option('ps_min_norm'  , 1e-20           , float       )
        o.ps_normalize  = conf.get_option('ps_normalize' , True            , conf.bool   )
        o.basisname     = conf.get_option('basisname'    , 'cc-pvqz-jkfit' , str         )
        o.coeff_order   = conf.get_option('coeff_order'  , 'pyscf'         , str         )
        o.overlap_order = conf.get_option('overlap_order', 'pyscf'         , str         )
        return o

    def get_all_paths():
        p = SimpleNamespace()
        p.xyzfilename      = conf.paths.get('xyzfile')
        p.basisfilename    = conf.paths.get('basisfile')
        p.chargefilename   = conf.paths.get('chargesfile')
        p.coefffilebase    = conf.paths.get('coeff_base')
        p.overfilebase     = conf.paths.get('over_base')

        p.splitpsfilebase  = conf.paths.get('ps_split_base')
        p.refsselfilebase  = conf.paths.get('refs_sel_base')
        p.powerrefbase     = conf.paths.get('ps_ref_base')

        p.kmmbase          = conf.paths.get('kmm_base')
        p.kernelconfbase   = conf.paths.get('kernel_conf_base')

        p.goodcoeffilebase = conf.paths.get('goodcoef_base')
        p.goodoverfilebase = conf.paths.get('goodover_base')
        p.baselinedwbase   = conf.paths.get('baselined_w_base')
        p.avfile           = conf.paths.get('averages_file')

        p.trainfilename    = conf.paths.get('trainingselfile')
        p.avecfilebase     = conf.paths.get('avec_base')
        p.bmatfilebase     = conf.paths.get('bmat_base')
        p.weightsfilebase  = conf.paths.get('weights_base')
        p.predictfilebase  = conf.paths.get('predict_base')
        p.outfilebase      = conf.paths.get('output_base')

        p.xyzexfilename    = conf.paths.get('ex_xyzfile')
        p.powerexbase      = conf.paths.get('ex_ps_base')
        p.kernelexbase     = conf.paths.get('ex_kernel_base')
        p.outexfilebase    = conf.paths.get('ex_output_base')
        return p

    path = get_config_path(argv)
    conf = Config(config_path=path)
    check_paths(conf)
    o = set_variable_values()
    p = get_all_paths()
    return o, p


def check_paths(conf):
    paths0 = [
      'xyzfile',
      'basisfile',
      'ex_xyzfile',
      'chargesfile',
      'averages_file',
      ]

    paths1 = [
      'avec_base',
      'baselined_w_base',
      'bmat_base',
      'coeff_base',
      'ex_kernel_base',
      'ex_output_base',
      'goodcoef_base',
      'goodover_base',
      'kernel_conf_base',
      'kmm_base',
      'output_base',
      'over_base',
      'predict_base',
      'ps_ref_base',
      'ps_split_base',
      'refs_sel_base',
      'qrefs_sel_base',
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
