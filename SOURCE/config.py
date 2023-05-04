import os.path
from types import SimpleNamespace
import configparser
import numpy as np

DEFAULT_PATH = 'config.txt'


class Config(object):
    def __init__(self, config_path=DEFAULT_PATH):
        if config_path==None:
            config_path = DEFAULT_PATH
        if not os.path.isfile(config_path):
            print("Cannot open configuration file \""+config_path+"\"")
            exit(1)
        print("Configuration file:", config_path)

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


def get_config_path(argv):
    path = None
    paths = ([x for x in argv[1:] if x.startswith('--config=')])
    if paths:
        path = paths[-1][len('--config='):]
    return path



def read_config(argv):
    def set_variable_values():
        o = SimpleNamespace()
        o.M = conf.get_option('m'           ,  100, int  )

        o.seed  = conf.get_option('seed'        ,    1, int  )
        o.train = conf.get_option('train_size'  , 1000, int  )
        o.fracs = conf.get_option('trainfrac', np.array([1.0]), conf.floats)

        o.soap_sigma = conf.get_option('soap_sigma'  ,  0.3, float  )
        o.soap_rcut  = conf.get_option('soap_rcut '  ,  4.0, float  )
        o.soap_ncut  = conf.get_option('soap_ncut '  ,  8  , int    )
        o.soap_lcut  = conf.get_option('soap_lcut '  ,  6  , int    )

        o.reorder_ao  = conf.get_option('reorder_ao'      ,  0, int)
        o.copy_metric = conf.get_option('copy_metric'     ,  1, int)

        o.reg  = conf.get_option('regular'  , 1e-6,            float)
        o.jit  = conf.get_option('jitter'   , 1e-10,           float)

        o.use_charges = conf.get_option('charges'  , 0,               int  )
        return o

    def get_all_paths():
        p = SimpleNamespace()
        p.xyzfilename      = conf.paths.get('xyzfile')
        p.basisfilename    = conf.paths.get('basisfile')
        p.kmmbase          = conf.paths.get('kmm_base')
        p.powerrefbase     = conf.paths.get('ps_ref_base')
        p.refsselfilebase  = conf.paths.get('refs_sel_base')
        p.kernelconfbase   = conf.paths['kernel_conf_base']
        p.splitpsfilebase  = conf.paths['ps_split_base']
        p.trainfilename    = conf.paths['trainingselfile']
        p.coefffilebase    = conf.paths['coeff_base']
        p.overfilebase     = conf.paths['over_base']
        p.goodcoeffilebase = conf.paths['goodcoef_base']
        p.goodoverfilebase = conf.paths['goodover_base']
        p.baselinedwbase   = conf.paths['baselined_w_base']
        p.avfile           = conf.paths['averages_file']
        p.xyzexfilename = conf.paths['ex_xyzfile']
        p.kernelexbase  = conf.paths['ex_kernel_base']
        p.powerexbase   = conf.paths['ex_ps_base']
        p.avecfilebase     = conf.paths['avec_base']
        p.bmatfilebase     = conf.paths['bmat_base']
        p.elselfilebase   = conf.paths['spec_sel_base']
        p.weightsfilebase = conf.paths['weights_base']
        p.predictfilebase = conf.paths['predict_base']
        p.chargefilename = conf.paths['chargesfile']
        return p

    path = get_config_path(argv)
    conf = Config(config_path=path)
    o = set_variable_values()
    p = get_all_paths()
    return o, p








