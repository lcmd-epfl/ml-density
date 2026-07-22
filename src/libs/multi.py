"""MPI wrappers.

MPI (mpi4py) is imported in each function, because the module should be able to be imported
and process_molecules() should work without MPI installed with use_mpi=False.
"""

import sys
import logging
from libs.progress import trange

logger = logging.getLogger('__main__')


def print_nodes(Nproc, nproc, comm):
    """Collect and log the processor name for each MPI rank.

    Args:
        Nproc (int): Total number of MPI processes.
        nproc (int): Current MPI rank.
        comm (mpi4py.MPI.Comm): Active MPI communicator.
    """
    from mpi4py import MPI
    sys.stdout.flush()
    msg = f'proc {nproc:3d} : {MPI.Get_processor_name()}'
    if nproc == 0:
        msg = [msg] + [comm.recv(source=i) for i in range(1, Nproc)]
        logger.debug('\n'.join(msg), extra={'flush': True})
    else:
        comm.send(msg, dest=0)
    comm.barrier()


def scatter_jobs(Nproc, nproc, comm, bra, ket, do_mol):
    """Distribute molecule indices to worker ranks and execute callbacks.

    Args:
        Nproc (int): Total number of MPI processes.
        nproc (int): Current MPI rank.
        comm (mpi4py.MPI.Comm): Active MPI communicator.
        bra (int): Inclusive start index.
        ket (int): Exclusive end index.
        do_mol (Callable[[int], None]): Worker callback run on each assigned molecule index.
    """
    from mpi4py import MPI
    if nproc == 0:
        for imol in range(bra, ket+Nproc-1):
            (npr, im) = comm.recv(source=MPI.ANY_SOURCE)
            im = imol if imol<ket else -1
            comm.send(im, dest=npr)
            logger.debug(f'sent {npr} : {im}', extra={'flush': True})
    else:
        imol = -1
        while True:
            comm.send((nproc, imol), dest=0)
            if (imol := comm.recv(source=0)) < 0:
                break
            do_mol(imol)
        logger.debug(f'{nproc} : finished', extra={'flush': True})


def multi_process(nmol, do_mol):
    """Execute per-molecule work either serially or with MPI job scattering.

    Args:
        nmol (int): Number of molecules to process.
        do_mol (Callable[[int], None]): Callback that processes one molecule index.
    """
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    Nproc = comm.Get_size()
    nproc = comm.Get_rank()

    print_nodes(Nproc, nproc, comm)

    if Nproc == 1:
        for imol in trange(nmol):
            do_mol(imol)
    else:
        scatter_jobs(Nproc, nproc, comm, 0, nmol, do_mol)


def process_molecules(nmol, do_mol, *, use_mpi=False):
    """Run per-molecule processing with consistent MPI/serial behavior.

    Args:
        nmol (int): Number of molecules to process.
        do_mol (Callable[[int], None]): Callback that processes one molecule index.
        use_mpi (bool): Whether to use MPI.
    """
    if use_mpi:
        multi_process(nmol, do_mol)
    else:
        for imol in trange(nmol):
            do_mol(imol)
