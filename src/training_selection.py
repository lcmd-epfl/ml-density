#!/usr/bin/env python3

import sys
import numpy as np
import ase.io
from libs.config import read_config


def main():
    o, p = read_config(sys.argv)
    np.random.seed(o.seed)
    nmol = len(ase.io.read(p.xyzfilename, ':'))
    train = np.random.choice(nmol, o.train, replace=False)
    np.savetxt(p.trainfilename, train, fmt='%i')


if __name__=='__main__':
    main()
