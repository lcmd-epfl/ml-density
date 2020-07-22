#!/usr/bin/env python3

import numpy as np
from config import Config
from ase.data import chemical_symbols
from functions import moldata_read,get_atomicindx,get_elements_list
from power_spectra_lib import reorder_ps


conf = Config()

def set_variable_values():
    m   = conf.get_option('m'           ,  100, int  )
    return [m]

[M] = set_variable_values()

xyzfilename     = conf.paths['xyzfile']
ps0file         = conf.paths['ps0file']
refsselfilebase = conf.paths['refs_sel_base']
elselfilebase   = conf.paths['spec_sel_base']

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

# number of molecules, number of atoms in each molecule, atomic numbers
(nmol, natoms, atomic_numbers) = moldata_read(xyzfilename)
natmax = max(natoms)
nenv = sum(natoms)

# elements array and atomic indices sorted by elements
elements = get_elements_list(atomic_numbers)
nel = len(elements)
(atomicindx, atom_counting, element_indices) = get_atomicindx(elements, atomic_numbers, natmax)

#====================== environmental power spectrum
power = np.load(ps0file)
nfeat = power.shape[-1]
power_env = np.zeros((nenv,nfeat),float)
ienv = 0
for imol in range(nmol):
    reorder_ps(power_env[ienv:ienv+natoms[imol]], power[imol], nel, atom_counting[imol], atomicindx[imol])
    ienv += natoms[imol]

ref_indices, distances = do_fps(power_env,M)
ref_elements = np.concatenate(element_indices)[ref_indices]

np.savetxt(refsselfilebase+str(M)+".txt",ref_indices,fmt='%i')
np.savetxt(elselfilebase+str(M)+".txt",ref_elements,fmt='%i')

for i,d in enumerate(distances):
  print(i+1, d)

el_count_total = sum(atom_counting)
el_count_ref = dict(zip( *np.unique(ref_elements, return_counts=True)))
for i in range(nel):
  n1 = el_count_ref[i]
  n2 = el_count_total[i]
  print('#', chemical_symbols[elements[i]]+':', n1, '/', n2, "(%.1f%%)"%(100.0*n1/n2) )

nuniq = len(np.unique(ref_indices))
if nuniq != len(ref_indices):
    print('warning: i have found only', nuniq, 'unique environments')

