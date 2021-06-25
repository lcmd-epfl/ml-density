#!/usr/bin/env python3

import sys
import numpy as np
from config import Config,get_config_path
from basis import basis_read
from functions import moldata_read,get_elements_list,get_atomicindx,print_progress
from power_spectra_lib import read_ps_1mol
from kernels_lib import kernel_nm_sparse_indices,kernel_nm

USEMPI = 1

path = get_config_path(sys.argv)
conf = Config(config_path=path)

def set_variable_values():
  m   = conf.get_option('m'           ,  100, int  )
  return [m]

[M] = set_variable_values()

xyzfilename     = conf.paths['xyzfile']
basisfilename   = conf.paths['basisfile']
elselfilebase   = conf.paths['spec_sel_base']
kernelconfbase  = conf.paths['kernel_conf_base']
kernelsizebase  = conf.paths['kernel_size_base']
powerrefbase    = conf.paths['ps_ref_base']
splitpsfilebase = conf.paths['ps_split_base']


(nmol, natoms, atomic_numbers) = moldata_read(xyzfilename)
natmax = max(natoms)

# elements array and atomic indices sorted by elements
elements = get_elements_list(atomic_numbers)
nel = len(elements)
(atomicindx, atom_counting, element_indices) = get_atomicindx(elements, atomic_numbers, natmax)

#====================================== reference environments
ref_elements = np.loadtxt(elselfilebase+str(M)+".txt",int)

# elements dictionary, max. angular momenta, number of radial channels
(el_dict, lmax, nmax) = basis_read(basisfilename)
if list(elements) != list(el_dict.values()):
    print("different elements in the molecules and in the basis:", list(elements), "and", list(el_dict.values()) )
    exit(1)
llmax = max(lmax.values())

# load power spectra
power_ref = {}
for l in range(llmax+1):
    power_ref[l] = np.load(powerrefbase+str(l)+"_"+str(M)+".npy");

#------------------------------------------------------------------------

def kernel_for_mol(imol):

    kernel_size, kernel_sparse_indices, kernel_size_indicators = kernel_nm_sparse_indices(M, natoms[imol], llmax, lmax, ref_elements, el_dict, atom_counting[imol])

    power_mol = {}
    for l in range(llmax+1):
        (dummy, power_mol[l]) = read_ps_1mol(splitpsfilebase+str(l)+"_"+str(imol)+".npy", nel, atom_counting[imol], atomicindx[imol])

    k_NM = kernel_nm(M, llmax, lmax, nel, el_dict, ref_elements, kernel_size, kernel_sparse_indices, power_mol, power_ref, atom_counting[imol], atomicindx[imol])
    np.savetxt(kernelconfbase+str(imol)+".dat", k_NM,fmt='%.06e')
    np.savetxt(kernelsizebase+str(imol)+".dat", kernel_size_indicators, fmt="%d")

#------------------------------------------------------------------------

if USEMPI==0:
  for imol in range(nmol):
    print_progress(imol, nmol)
    kernel_for_mol(imol)

else:
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  Nproc = comm.Get_size()
  nproc = comm.Get_rank()

  msg = "proc " + "%3d"%(nproc) + ' : ' + MPI.Get_processor_name()
  if nproc == 0:
    print(msg)
    for i in range(1, Nproc):
      msg = comm.recv(source=i)
      print (msg)
    print(flush=1)
  else:
    comm.send(msg, dest=0)
  comm.barrier()

  if Nproc == 1:
    for imol in range(nmol):
      print_progress(imol, nmol)
      kernel_for_mol(imol)

  else:

    if nproc == 0:
      for imol in range(nmol+Nproc-1):
        (npr, im) = comm.recv(source=MPI.ANY_SOURCE)
        im = imol if imol<nmol else -1
        comm.send(im, dest=npr);
        print("sent", npr, ':', im, flush=1)

    else:
      imol = -1
      while True:
        comm.send((nproc, imol), dest=0)
        imol = comm.recv(source=0)
        if(imol<0):
          break
        kernel_for_mol(imol)
      print(nproc, ':', 'finished')

