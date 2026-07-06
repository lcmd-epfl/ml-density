import logging
import numpy as np
import metatensor
from libs.get_matrices_A import print_batches
from libs.multi import print_nodes, scatter_jobs

logger = logging.getLogger('__main__')


def print_mem(totsize, ntrain):
    b2mib = 1.0/(1<<20)
    b2gib = 1.0/(1<<30)
    size = symsize(totsize)*np.array(0.0).itemsize
    logger.info(f"""Problem dimensionality = {totsize}\n\
Number of training molecules = {ntrain}\n\
output: {size:16d} bytes ({size*b2mib:10.2f} MiB, {size*b2gib:6.2f} GiB)\n""", extra={'flush': True})
    return


def symsize(M):
    return (M*(M+1))//2


def mpos(i, j):
    # A[i+j*(j+1)/2], i <= j, 0 <= j < N
    return i + (((j)*((j)+1))//2)


def do_work_b(idx, nmax, conf, ref_elem, path_over, path_kern, Bmat):

    over = metatensor.load(path_over.format(conf))
    k_NM = metatensor.load(path_kern.format(conf))

    for (l1, l2, q1, q2), oblock in over.items():
        msize1 = 2*l1+1
        msize2 = 2*l2+1
        nsize1 = nmax[q1][l1]
        nsize2 = nmax[q2][l2]
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


def get_b(basis, ref_elem, ntrains, trrange,
          path_over, path_kern, paths_bmat, use_mpi):

    def do_mol(imol):
        do_work_b(idx, basis.nmax, trrange[imol], ref_elem, path_over, path_kern, Bmat)

    totsize = basis.nao_for_mol(ref_elem)
    Bmat = np.zeros(matsize := symsize(totsize))
    idx = basis.sparse_indices(ref_elem)

    if use_mpi:
        from mpi4py import MPI
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
        print_batches(ntrains, paths_bmat)
    if use_mpi:
        MPI.COMM_WORLD.barrier()

    if Nproc==1:
        for ifrac, path_bmat in enumerate(paths_bmat):
            for imol in range(ntrains[ifrac-1], ntrains[ifrac]):
                logger.info(f'{nproc:4d}: {imol:4d}', extra={'flush': True})
                do_mol(imol)
            Bmat.tofile(path_bmat)
        if use_mpi:
            t = MPI.Wtime () - t
            logger.info(f'{t=:4.2f}', extra={'flush': True})

    else:
        bufsize = (1<<30)//np.array(0.0).itemsize  # number of doubles to take 1 GiB
        if bufsize > matsize:
            bufsize = matsize
        div = matsize//bufsize
        rem = matsize%bufsize
        if nproc==0:
            BMAT = np.zeros(bufsize)

        for ifrac, path_bmat in enumerate(paths_bmat):
            scatter_jobs(Nproc, nproc, MPI.COMM_WORLD, ntrains[ifrac-1], ntrains[ifrac], do_mol)
            MPI.COMM_WORLD.barrier()

            if nproc==0:
                tt = MPI.Wtime()
                logger.info(f'batch{ifrac}: t={tt-t:4.2f}', extra={'flush': True})
                t = tt
            for i in range(div+1):
                if (size := bufsize if i<div else rem)==0:
                    break
                MPI.COMM_WORLD.Reduce(Bmat[i*bufsize:i*bufsize+size], BMAT[:size] if nproc==0 else None, MPI.SUM, 0)
                if nproc==0:
                    logger.info(f'chunk #{i+1}/{div+1 if rem else div} written', extra={'flush': True})
                    with open(path_bmat, 'a' if i else 'w') as f:
                        BMAT[:size].tofile(f)
