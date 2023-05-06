#!/usr/bin/env python3

import sys
import os.path
from config import Config, get_config_path


def main():
    path = get_config_path(sys.argv)
    conf = Config(config_path=path)

    paths0 = [
      'xyzfile',
      'basisfile',
      'ex_xyzfile',
      'chargesfile',
      'averages_file'
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
      'spec_sel_base',
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
            print(f'Cannot find directory "{path}" ("{key}")')
            dirs.append(path)
      else:
        print(f'Cannot find option "{key}"')
    print()

    for d in sorted(set(dirs)):
        print(f'mkdir -p {d}')


if __name__=='__main__':
    main()
