import sys
import logging
from tqdm import trange
from mpi4py import MPI

logger = logging.getLogger('__main__')


def print_nodes(Nproc, nproc, comm):
    sys.stdout.flush()
    msg = f'proc {nproc:3d} : {MPI.Get_processor_name()}'
    if nproc == 0:
        msg = [msg] + [comm.recv(source=i) for i in range(1, Nproc)]
        logger.info('\n'.join(msg), extra={'flush': True})
    else:
        comm.send(msg, dest=0)
    comm.barrier()


def scatter_jobs(Nproc, nproc, comm, bra, ket, do_mol):
    if nproc == 0:
        for imol in range(bra, ket+Nproc-1):
            (npr, im) = comm.recv(source=MPI.ANY_SOURCE)
            im = imol if imol<ket else -1
            comm.send(im, dest=npr)
            logger.info(f'sent {npr} : {im}', extra={'flush': True})
    else:
        imol = -1
        while True:
            comm.send((nproc, imol), dest=0)
            if (imol := comm.recv(source=0)) < 0:
                break
            do_mol(imol)
        logger.info(f'{nproc} : finished', extra={'flush': True})


def multi_process(nmol, do_mol):
    comm = MPI.COMM_WORLD
    Nproc = comm.Get_size()
    nproc = comm.Get_rank()

    print_nodes(Nproc, nproc, comm)

    if Nproc == 1:
        for imol in trange(nmol):
            do_mol(imol)
    else:
        scatter_jobs(Nproc, nproc, comm, 0, nmol, do_mol)
