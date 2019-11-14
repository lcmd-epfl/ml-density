#!/usr/bin/python3

import numpy as np
from config import Config

conf = Config()

def set_variable_values():
    f  = conf.get_option('trainfrac'   ,  1.0,   float)
    m  = conf.get_option('m'           ,  100,   int  )
    r  = conf.get_option('regular'     ,  1e-6,  float)
    j  = conf.get_option('jitter'      ,  1e-10, float)
    return [f,m,r,j]

[frac,M,reg,jit] = set_variable_values()

kmmbase         = conf.paths['kmm_base']
avecfilebase    = conf.paths['avec_base']
bmatfilebase    = conf.paths['bmat_base']
weightsfilebase = conf.paths['weights_base']

Avec = np.loadtxt(avecfilebase + "_M"+str(M)+"_trainfrac"+str(frac)+".txt")
Bmat = np.loadtxt(bmatfilebase + "_M"+str(M)+"_trainfrac"+str(frac)+".txt")
Rmat = np.load(kmmbase+str(M)+".npy")

totsize = Avec.shape[0]
print("problem dimensionality =", totsize)

weights = np.linalg.solve(Bmat + reg*Rmat + jit*np.eye(totsize),Avec)
np.save(weightsfilebase + "_M"+str(M)+"_trainfrac"+str(frac)+"_reg"+str(reg)+"_jit"+str(jit)+".npy",weights)

