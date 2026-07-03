#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import ase.io
from libs.config import read_config


def main():
    o, p = read_config(sys.argv)
    np.random.seed(o.seed)
    nmol = len(ase.io.read(p.xyzfilename, ':'))

    train = np.random.choice(nmol, o.train, replace=False)
    test = np.setdiff1d(range(nmol), train, assume_unique=True)

    sets = pd.DataFrame({'mol_idx': np.hstack((train, test)),
                         'subset': ['train'] * o.train + ['test'] * (nmol-o.train),
                         })
    sets.to_csv(p.train_test_sets, index=False)


if __name__=='__main__':
    main()
