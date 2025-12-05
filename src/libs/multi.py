import sys
from mpi4py import MPI
from libs.functions import print_progress


def print_nodes(Nproc, nproc, comm):
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


def scatter_jobs(Nproc, nproc, comm, bra, ket, do_mol):
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


def multi_process(nmol, do_mol):
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
