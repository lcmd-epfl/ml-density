#!/usr/bin/env python3

import sys
import os.path
from config import Config,get_config_path

path = get_config_path(sys.argv)
conf = Config(config_path=path)

paths0 = [
  'xyzfile',
  'basisfile',
  'ex_xyzfile',
  'ps0file',
  'chargesfile',
  ]

paths1 = [
  'avec_base',
  'averages_dir',
  'baselined_w_base',
  'bmat_base',
  'coeff_base',
  'ex_kernel_base',
  'ex_output_base',
  'ex_predict_base',
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
  'spec_sel_base',
  'weights_base',
  'trainingselfile',
  'ex_ps_base',
  'charges_base',
  'kernel_charges_base',
  'pca_dir',
  ]

for key in paths0:
  if key in conf.paths:
    path = conf.paths[key]
    isfile = os.path.isfile(path)
    if not isfile:
      print("Cannot find file '%s' ('%s')" % (path, key))
  else:
    print("Cannot find option '%s'" % (key))

dirs = []

for key in paths1:
  if key in conf.paths:
    path = conf.paths[key]
    if not key.endswith('_dir'):
      path = os.path.dirname(path)
    isdir = os.path.isdir(path)
    if not isdir:
      print("Cannot find directory '%s' ('%s')" % (path, key))
      dirs.append(path)
  else:
    print("Cannot find option '%s'" % (key))

print()
for d in sorted(set(dirs)):
  print ('mkdir -p', d)


