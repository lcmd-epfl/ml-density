#!/usr/bin/env python3

import sys
import numpy as np
from config import Config,get_config_path
from ase.data import chemical_symbols
from functions import moldata_read,get_atomicindx,get_elements_list
from power_spectra_lib import reorder_ps, read_ps_1mol

path = get_config_path(sys.argv)
conf = Config(config_path=path)

def set_variable_values():
    m   = conf.get_option('m'           ,  100, int  )
    return [m]

[M] = set_variable_values()

xyzfilename     = conf.paths['xyzfile']
ps0file         = conf.paths['ps0file']
splitpsfilebase = conf.paths['ps_split_base']
refsselfilebase = conf.paths['refs_sel_base']
elselfilebase   = conf.paths['spec_sel_base']
pcadir          = conf.paths['pca_dir']

def do_fps(x, d=0):
    # Code from Giulio Imbalzano
    n = len(x)
    if d==0:
        d = n
    iy = np.zeros(d,int)
    measure = np.zeros(d-1,float)
    iy[0] = 0
    # Faster evaluation of Euclidean distance
    n2 = np.sum(x*x, axis=1)
    dl = n2 + n2[iy[0]] - 2.0*np.dot(x,x[iy[0]])
    for i in range(1,d):
        iy[i], measure[i-1] = np.argmax(dl), np.amax(dl)
        nd = n2 + n2[iy[i]] - 2.0*np.dot(x,x[iy[i]])
        dl = np.minimum(dl,nd)
    return iy, measure

def svd(nel, power_env, ref_indices, ref_elements):
    lambda2 = []
    weights = []
    for iq in range(nel):
        ridx  = ref_indices[np.where(ref_elements==iq)]
        A     = power_env[ridx]
        A2    = A@A.T
        l2, u = np.linalg.eigh(A2)   # PCA of A.T
        idx   = l2.argsort()[::-1]
        l2    = l2[idx]
        u     = u[:,idx]
        lut   = (u / np.sqrt(l2)).T  # to save the norm
        #A_new = lut @ A
        weights.append(lut)
        lambda2.append(l2)
    return lambda2, weights

# number of molecules, number of atoms in each molecule, atomic numbers
(nmol, natoms, atomic_numbers) = moldata_read(xyzfilename)
natmax = max(natoms)
nenv = sum(natoms)

# elements array and atomic indices sorted by elements
elements = get_elements_list(atomic_numbers)
nel = len(elements)
(atomicindx, atom_counting, element_indices) = get_atomicindx(elements, atomic_numbers, natmax)
element_indices = np.concatenate(element_indices)

#====================== environmental power spectrum
try:
    power = np.load(ps0file)
    nfeat = power.shape[-1]
    power_env = np.zeros((nenv,nfeat),float)
    ienv = 0
    for imol in range(nmol):
        reorder_ps(power_env[ienv:ienv+natoms[imol]], power[imol], nel, atom_counting[imol], atomicindx[imol])
        ienv += natoms[imol]
except:
    power_env = np.vstack([read_ps_1mol(f"{splitpsfilebase}0_{imol}.npy", nel, atom_counting[imol], atomicindx[imol])[1] for imol in range(nmol)])



ref_indices, distances = do_fps(power_env, M)
ref_elements = element_indices[ref_indices]

for i,d in enumerate(distances):
    print(i+1, d)

el_count_total = sum(atom_counting)
el_count_ref = dict(zip( *np.unique(ref_elements, return_counts=True)))
el_count_ref = [el_count_ref[i] for i in range(nel)]
for i in range(nel):
    n1 = el_count_ref[i]
    n2 = el_count_total[i]
    print('#', chemical_symbols[elements[i]]+':', n1, '/', n2, "(%.1f%%)"%(100.0*n1/n2) )

nuniq = len(np.unique(ref_indices))
if nuniq != len(ref_indices):
    print(f'warning: I have found only {nuniq} unique environments')


lambda2, weights = svd(nel, power_env, ref_indices, ref_elements)
for iq in range(nel):
    symb  = chemical_symbols[elements[iq]]
    print(symb, lambda2[iq])
    np.save(f'{pcadir}/{symb}.npy', weights[iq])


np.savetxt(refsselfilebase+str(M)+".txt",ref_indices,fmt='%i')
np.savetxt(elselfilebase+str(M)+".txt",ref_elements,fmt='%i')
