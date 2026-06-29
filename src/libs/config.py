#!/usr/bin/env python3
'''Read configuration file and return options and paths [o, p =read_config(argv)].'''
import sys
import os, os.path
from types import SimpleNamespace
import configparser
import numpy as np

DEFAULT_PATH = 'config.txt'


class Config(object):
    def __init__(self, config_path: str = DEFAULT_PATH) -> None:
        if config_path==None:
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

    def get_option(self, key: str, default, ttype: type):
        '''Return option value cast to ttype, or default if key is absent.'''
        if key in self.options:
            return ttype(self.options[key])
        else:
            return default

    def floats(self, x: str) -> np.ndarray:
        '''Parse a comma-separated string into a sorted array of unique floats.'''
        return np.unique(list(map(float, x.split(','))))

    def bool(self, x: str) -> bool:
        '''Parse a string to boolean (accepts 0/1/true/false).'''
        x = x.lower()
        if x in ['1', 'true']:
            return True
        elif x in ['0', 'false']:
            return False
        print(f'Wrong input for the Bool option: "{x}"')
        exit(0)


def get_config_path(argv: list[str]) -> str | None:
    '''Extract --config=<path> from command-line arguments.'''
    path = None
    paths = ([x for x in argv[1:] if x.startswith('--config=')])
    if paths:
        path = paths[-1][len('--config='):]
    return path


def read_config(argv: list[str]) -> tuple[SimpleNamespace, SimpleNamespace]:
    '''Parse configuration file and return options (o) and paths (p).'''
    def set_variable_values():
        o = SimpleNamespace()
        # Number of sparse reference environments for kernel approximation
        o.M            = conf.get_option('m'           , 100             , int         )
        # Random seed for reproducibility of FPS selection and train/test splits
        o.seed         = conf.get_option('seed'        , 1               , int         )
        # Number of training structures used for regression
        o.train        = conf.get_option('train_size'  , 1000            , int         )
        # Fractions of the training set to use (for learning curves)
        o.fracs        = conf.get_option('trainfrac'   , np.array([1.0]) , conf.floats )
        # SOAP Gaussian smearing width [Angstrom]: controls the broadening of atomic densities
        o.soap_sigma   = conf.get_option('soap_sigma'  , 0.3             , float       )
        # SOAP radial cutoff [Angstrom]: defines the size of the atomic environment
        o.soap_rcut    = conf.get_option('soap_rcut'   , 4.0             , float       )
        # SOAP maximum number of radial basis functions: controls radial resolution
        o.soap_ncut    = conf.get_option('soap_ncut'   , 8               , int         )
        # SOAP maximum angular momentum l: controls angular resolution of the expansion
        o.soap_lcut    = conf.get_option('soap_lcut'   , 6               , int         )
        # Reorder atomic orbitals to match a target convention (e.g. PySCF vs Gaussian)
        o.reorder_ao   = conf.get_option('reorder_ao'  , 0               , int         )
        # Copy the overlap matrix S as the metric for the density fitting coefficients
        o.copy_metric  = conf.get_option('copy_metric' , 1               , int         )
        # Regularization: added as reg*K_MM (B + reg*K_MM + jit*I), controls overfitting
        o.reg          = conf.get_option('regular'     , 1e-6            , float       )
        # Jitter: added as jit*I = (B + reg*K_MM + jit*I), ensures numerical stability
        o.jit          = conf.get_option('jitter'      , 1e-10           , float       )
        # Include nuclear charges as additional features in the descriptor
        o.use_charges  = conf.get_option('charges'     , 0               , int         )
        # Minimum norm threshold: power spectrum components below this are zeroed out
        o.ps_min_norm  = conf.get_option('ps_min_norm' , 1e-20           , float       )
        # Normalize each power spectrum vector to unit norm
        o.ps_normalize = conf.get_option('ps_normalize', True            , conf.bool   )
        return o

    def get_all_paths():
        p = SimpleNamespace()
        # Molecular geometries in XYZ format (16.xyz)
        p.xyzfilename      = conf.paths.get('xyzfile')
        # Basis set definition l and n per element (cc-pvqz-jkfit.1.d2k)
        p.basisfilename    = conf.paths.get('basisfile')
        # Molecule charge file (charges.dat)
        p.chargefilename   = conf.paths.get('chargesfile')
        # Auxiliary basis coefficients c from an ab initio calculation (ALL_C/mol_{imol}.dat)
        p.coefffilebase    = conf.paths.get('coeff_base')
        # Overlap matrices S of the auxiliary basis functions (ALL_J/mol_{imol}.npy)
        p.overfilebase     = conf.paths.get('over_base')

        # Per-molecule SOAP power spectra split by angular channel l (INNER/PS/PS_{imol}.mts)
        p.splitpsfilebase  = conf.paths.get('ps_split_base')
        # Farthest Point Sampling(FPS)-selected sparse reference environments
        # (INNER/SELECTIONS/refs_selection_{M}.txt, molecule index + element)
        p.refsselfilebase  = conf.paths.get('refs_sel_base')
        # Power spectra of the M sparse reference environments (INNER/PS_{M}.mts)
        p.powerrefbase     = conf.paths.get('ps_ref_base')

        # Reference-reference kernel matrix K_MM (INNER/KMM{M}.mts)
        p.kmmbase          = conf.paths.get('kmm_base')
        # Per-molecule kernel vectors K_NM between training configs and references (INNER/KERNELS/kernel_conf{imol}.mts)
        p.kernelconfbase   = conf.paths.get('kernel_conf_base')

        # Coefficients projected onto the symmetry-adapted basis (INNER/coeff/mol_{imol}.npy)
        p.goodcoeffilebase = conf.paths.get('goodcoef_base')
        # Overlap matrices projected onto the symmetry-adapted basis (INNER/metric/mol_{imol}.mts)
        p.goodoverfilebase = conf.paths.get('goodover_base')
        # Baselined density fitting weights, coefficients minus per-element averages (INNER/BASELINED_PROJECTIONS/projections_conf{imol}.mts)
        p.baselinedwbase   = conf.paths.get('baselined_w_base')
        # Per-element average density coefficients, the baseline (INNER/AVERAGES.mts)
        p.avfile           = conf.paths.get('averages_file')

        # List of molecule indices selected for the training set (INNER/SELECTIONS/training_selection.txt)
        p.trainfilename    = conf.paths.get('trainingselfile')
        # Target projection A = sum_n K_NM^T w_n, density projected into kernel space (INNER/Avec_M{M}_trainfrac{frac}.txt)
        p.avecfilebase     = conf.paths.get('avec_base')
        # Gram matrix B = sum_n K_NM^T S_n K_NM, overlap-weighted kernel Gram matrix (INNER/Bmat_M{M}_trainfrac{frac}.dat)
        p.bmatfilebase     = conf.paths.get('bmat_base')
        # Regression weights from solving the KRR system (INNER/weights_M{M}_trainfrac{frac}_reg{reg}_jit{jit}.npy)
        p.weightsfilebase  = conf.paths.get('weights_base')
        # Predicted density coefficients in the symmetry-adapted basis (INNER/prediction_test_M{M}_trainfrac{frac}_reg{reg}_jit{jit}.mts)
        p.predictfilebase  = conf.paths.get('predict_base')
        # Final output: predicted density coefficients in the original AO basis (INNER/predicted/rho_{imol}.dat)
        p.outfilebase      = conf.paths.get('output_base')

        # Molecular geometries for extrapolation, out-of-sample molecules (extra/1.xyz)
        p.xyzexfilename    = conf.paths.get('ex_xyzfile')
        # SOAP power spectra for the extrapolation molecules (extra/PS_{imol}.mts)
        p.powerexbase      = conf.paths.get('ex_ps_base')
        # Kernel vectors between extrapolation molecules and sparse references (extra/kernel{imol}.mts)
        p.kernelexbase     = conf.paths.get('ex_kernel_base')
        # Predicted density coefficients for the extrapolation molecules (extra/rho_{imol}.dat)
        p.outexfilebase    = conf.paths.get('ex_output_base')
        return p

    path = get_config_path(argv)
    if path is None:
        path = DEFAULT_PATH
        if not os.path.isfile(path):
            print(f'Cannot find configuration file "{path}"')
            exit(1)
    conf = Config(config_path=path)
    check_paths(conf)
    o = set_variable_values()
    p = get_all_paths()
    return o, p


def check_paths(conf: Config) -> None:
    '''Warn about missing input files and create missing output directories.'''
    # Input files that must already exist on disk
    input_files = [
      'xyzfile',
      'basisfile',
      'ex_xyzfile',
      'chargesfile',
      'averages_file'
      ]

    # Output file bases whose parent directories will be created if missing
    output_bases = [
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

    # Check the existence of all input files
    for key in input_files:
        if key in conf.paths:
            path = conf.paths[key]
            if not os.path.isfile(path):
                print(f'Cannot find file "{path}" ("{key}")')
        else:
            print(f'Cannot find option "{key}"')

    # Check the existence of all output directories and create them if missing
    dirs = set()
    for key in output_bases:
      if key in conf.paths:
          path = conf.paths[key]
          path = os.path.dirname(path)
          if not os.path.isdir(path) and path not in dirs:
            dirs.add(path)
      else:
        print(f'Cannot find option "{key}"')
    print()

    for d in sorted(dirs):
        print(f'Creating directory {d}')
        os.makedirs(d, exist_ok=True)
