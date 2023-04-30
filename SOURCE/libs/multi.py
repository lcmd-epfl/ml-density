from mpi4py import MPI
from functions import print_progress

def multi_process(nmol, do_mol):
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
            do_mol(imol)
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
                do_mol(imol)
            print(f'{nproc} : finished')
