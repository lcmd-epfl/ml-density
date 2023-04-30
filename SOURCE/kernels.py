#!/usr/bin/env python3

import sys
import numpy as np
import equistore
from config import Config,get_config_path
from basis import basis_read
from functions import moldata_read, get_elements_list, get_atomicindx, print_progress
from libs.kernels_lib import kernel_nm_sparse_indices, kernel_nm_new

USEMPI = 1


def set_variable_values(conf):
    m   = conf.get_option('m'           ,  100, int  )
    return [m]

def main():
    path = get_config_path(sys.argv)
    conf = Config(config_path=path)
    [M] = set_variable_values(conf)
    xyzfilename     = conf.paths['xyzfile']
    basisfilename   = conf.paths['basisfile']
    elselfilebase   = conf.paths['spec_sel_base']
    kernelconfbase  = conf.paths['kernel_conf_base']
    powerrefbase    = conf.paths['ps_ref_base']
    splitpsfilebase = conf.paths['ps_split_base']


    def kernel_for_mol(imol):
        kernel_size, kernel_sparse_indices = kernel_nm_sparse_indices(M, natoms[imol], llmax, lmax,
                                                                      ref_elements, el_dict, atom_counting[imol])
        power_mol_new = equistore.load(f'{splitpsfilebase}_{imol}.npz')
        k_NM = kernel_nm_new(lmax, nel, el_dict, ref_elements, kernel_size, kernel_sparse_indices,
                             power_mol_new, power_ref_new, atom_counting[imol], atomicindx[imol])
        np.savetxt(f'{kernelconfbase}{imol}.dat', k_NM, fmt='%.06e')

    def single_process():
        for imol in range(nmol):
            print_progress(imol, nmol)
            kernel_for_mol(imol)

    def multi_process():
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        Nproc = comm.Get_size()
        nproc = comm.Get_rank()
        msg = f'proc {nproc:3d} : {MPI.Get_processor_name()}'
        if nproc == 0:
            print(msg)
            for i in range(1, Nproc):
                msg = comm.recv(source=i)
                print(msg)
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
                    print(f'sent {npr} : {im}', flush=1)
            else:
                imol = -1
                while True:
                    comm.send((nproc, imol), dest=0)
                    imol = comm.recv(source=0)
                    if(imol<0):
                        break
                    kernel_for_mol(imol)
                print(f'{nproc} : finished')


    (nmol, natoms, atomic_numbers) = moldata_read(xyzfilename)
    natmax = max(natoms)

    # elements array and atomic indices sorted by elements
    elements = get_elements_list(atomic_numbers)
    nel = len(elements)
    atomicindx, atom_counting, _ = get_atomicindx(elements, atomic_numbers, natmax)

    ref_elements = np.loadtxt(f'{elselfilebase}{M}.txt', dtype=int)
    power_ref_new = equistore.load(f'{powerrefbase}_{M}.npz');

    (el_dict, lmax, nmax) = basis_read(basisfilename)
    if list(elements) != list(el_dict.values()):
        print("different elements in the molecules and in the basis:", list(elements), "and", list(el_dict.values()) )
        exit(1)
    llmax = max(lmax.values())


    if USEMPI==0:
        single_process()
    else:
        multi_process()


if __name__=='__main__':
    main()
