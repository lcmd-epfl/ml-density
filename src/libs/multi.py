import sys
from mpi4py import MPI
from libs.functions import print_progress


def print_nodes(Nproc: int, nproc: int, comm: MPI.Comm) -> None:
    '''Print the hostname of each MPI rank.'''
    sys.stdout.flush()
    msg = f'proc {nproc:3d} : {MPI.Get_processor_name()}'
    if nproc == 0:
        print(msg)
        for i in range(1, Nproc):
            msg = comm.recv(source=i)
            print(msg)
        print(flush=True)
    else:
        comm.send(msg, dest=0)
    comm.barrier()


def scatter_jobs(Nproc: int, nproc: int, comm: MPI.Comm, bra: int, ket: int, do_mol: callable) -> None:
    '''Distribute molecule indices [bra, ket) across MPI ranks.'''
    if nproc == 0:
        for imol in range(bra, ket+Nproc-1):
            (npr, im) = comm.recv(source=MPI.ANY_SOURCE)
            im = imol if imol<ket else -1
            comm.send(im, dest=npr);
            print(f'sent {npr} : {im}', flush=True)
    else:
        imol = -1
        while True:
            comm.send((nproc, imol), dest=0)
            imol = comm.recv(source=0)
            if(imol<0):
                break
            do_mol(imol)
        print(f'{nproc} : finished', flush=True)


def multi_process(nmol: int, do_mol: callable) -> None:
    '''Run do_mol over nmol molecules, parallelized with MPI if available.'''
    comm = MPI.COMM_WORLD
    Nproc = comm.Get_size()
    nproc = comm.Get_rank()

    print_nodes(Nproc, nproc, comm)

    if Nproc == 1:
        for imol in range(nmol):
            print_progress(imol, nmol)
            do_mol(imol)
    else:
        scatter_jobs(Nproc, nproc, comm, 0, nmol, do_mol)
