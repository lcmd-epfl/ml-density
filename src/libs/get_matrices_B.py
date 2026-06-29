import numpy as np
import metatensor
from libs.tmap import sparseindices_fill
from libs.get_matrices_A import print_batches
from libs.multi import print_nodes, scatter_jobs
USE_MPI = 1
if USE_MPI:
    from mpi4py import MPI


def print_mem(totsize: int, ntrain: int) -> None:
    '''Print memory estimate for the Gram matrix.'''
    b2mib = 1.0/(1<<20)
    b2gib = 1.0/(1<<30)
    size = symsize(totsize)*np.array(0.0).itemsize
    print(f"""\
        Problem dimensionality = {totsize}\n\
        Number of training molecules = {ntrain}\n\
        output: {size:16d} bytes ({size*b2mib:10.2f} MiB, {size*b2gib:6.2f} GiB)\n""", flush=True)
    return


def symsize(M: int) -> int:
    '''Return packed size of an M x M symmetric matrix.'''
    return (M*(M+1))>>1


def mpos(i: int, j: int) -> int:
    '''Return packed index for element (i, j) of a symmetric matrix, i <= j.'''
    return (i)+(((j)*((j)+1))>>1)


def do_work_b(idx: np.ndarray, nmax: dict, conf: int, ref_elem: np.ndarray, path_over: str, path_kern: str, Bmat: np.ndarray) -> None:
    '''Accumulate one molecule's contribution to the Gram matrix B = sum_n K_NM^T S_n K_NM.'''
    over = metatensor.load(f'{path_over}{conf}.mts')
    k_NM = metatensor.load(f'{path_kern}{conf}.mts')

    for (l1, l2, q1, q2), oblock in over.items():
        msize1 = 2*l1+1
        msize2 = 2*l2+1
        nsize1 = nmax[(q1,l1)]
        nsize2 = nmax[(q2,l2)]
        kblock1 = k_NM.block(o3_lambda=l1, center_type=q1)
        kblock2 = k_NM.block(o3_lambda=l2, center_type=q2)
        oval = oblock.values.reshape(len(kblock1.samples), len(kblock2.samples), msize1, msize2, nsize1, nsize2)

        for iiref1, iref1 in enumerate(np.where(ref_elem==q1)[0]):
            for iiref2, iref2 in enumerate(np.where(ref_elem==q2)[0]):
                if iref1>iref2:
                    continue
                # dB = np.einsum('AMJ,AaMmNn,amj->NnJj', kblock1.values[...,iiref1], oval, kblock2.values[...,iiref2])
                t1 = np.einsum('AMJ,AaMmNn->amJNn', kblock1.values[...,iiref1], oval)
                dB = np.einsum('amJNn,amj->njNJ', t1, kblock2.values[...,iiref2])

                i1 = idx[iref1, l1]
                for n2 in range(nsize2):
                    for im2 in range(msize2):
                       i2 = idx[iref2, l2]+ n2*msize2 + im2
                       i12 = mpos(i1,i2)
                       if (iref1!=iref2) or (iref1==iref2 and l1<l2):
                            Bmat[i12:i12+msize1*nsize1] += dB[n2,im2,:,:].flatten()
                       elif iref1==iref2 and l1==l2:
                            i12a = i12 + msize1*n2
                            i12b = i12a + im2+1
                            Bmat[i12:i12a]  += dB[n2,im2,:n2,:].flatten()
                            Bmat[i12a:i12b] += dB[n2,im2,n2,:im2+1]


def get_b(lmax: dict, nmax: dict, totsize: int, ref_elem: np.ndarray, nfrac: int, ntrains: np.ndarray, trrange: np.ndarray,
          path_over: str, path_kern: str, paths_bmat: list[str]) -> None:
    '''Build the Gram matrix B = sum_n K_NM^T S_n K_NM and save to disk.'''

    def do_mol(imol):
        do_work_b(idx, nmax, trrange[imol], ref_elem, path_over, path_kern, Bmat)

    Bmat = np.zeros(symsize(totsize))
    idx = sparseindices_fill(lmax, nmax, ref_elem)
    ntrains = np.pad(ntrains, (0, 1), 'constant', constant_values=0)

    if USE_MPI:
        Nproc = MPI.COMM_WORLD.Get_size()
        nproc = MPI.COMM_WORLD.Get_rank()
        print_nodes(Nproc, nproc, MPI.COMM_WORLD)
        t = 0.0
        if nproc==0:
            print_mem(totsize, ntrains[-2])
            t = MPI.Wtime()
        MPI.COMM_WORLD.barrier()
    else:
        nproc = 0
        Nproc = 1

    if nproc==0:
        print_batches(nfrac, ntrains, paths_bmat)
    if USE_MPI:
        MPI.COMM_WORLD.barrier()

    if Nproc==1:
        for ifrac in range(nfrac):
            for imol in range(ntrains[ifrac-1], ntrains[ifrac]):
                print(f'{nproc:4d}: {imol:4d}', flush=True)
                do_mol(imol)
            Bmat.tofile(paths_bmat[ifrac])
        if USE_MPI:
            t = MPI.Wtime () - t
            print(f'{t=:4.2f}', flush=True)

    else:
        bufsize = (1<<30)//np.array(0.0).itemsize  # number of doubles to take 1 GiB
        if bufsize > symsize(totsize):
            bufsize = symsize(totsize)
        div = symsize(totsize)//bufsize
        rem = symsize(totsize)%bufsize
        if nproc==0:
            BMAT = np.zeros(bufsize)

        for ifrac in range(nfrac):
            scatter_jobs(Nproc, nproc, MPI.COMM_WORLD, ntrains[ifrac-1], ntrains[ifrac], do_mol)
            MPI.COMM_WORLD.barrier()

            if nproc==0:
                tt = MPI.Wtime()
                print(f'batch{ifrac}: t={tt-t:4.2f}', flush=True)
                t = tt
            for i in range(div+1):
                size = bufsize if i<div else rem
                if size==0:
                    break
                MPI.COMM_WORLD.Reduce(Bmat[i*bufsize:i*bufsize+size], BMAT[:size] if nproc==0 else None, MPI.SUM, 0)
                if nproc==0:
                    print(f'chunk #{i+1}/{div+1 if rem else div} written', flush=True)
                    with open(paths_bmat[ifrac], 'a' if i else 'w') as f:
                        BMAT[:size].tofile(f)
