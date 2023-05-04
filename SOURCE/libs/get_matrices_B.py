import sys
import numpy as np
import equistore
from libs.tmap import vector2tmap, tmap2vector

def symsize(M):
    return (M*(M+1))>>1;

def mpos(i, j):
    # A[i+j*(j+1)/2], i <= j, 0 <= j < N
    return (i)+(((j)*((j)+1))>>1);

USE_MPI=0
Nproc = 1
nproc = 0

def print_batches(f, nfrac, ntrains, paths):
    if nproc==0:
        for i in range(nfrac):
           print(f"batch {i:2d} [{ntrains[i-1]}--{ntrains[i]}):\t {paths[i]}", file=f);
        print(flush=True, file=f)
    if USE_MPI:
        MPI_Barrier(MPI_COMM_WORLD);


def sparseindices_fill(lmax, nmax, atoms):
    idx = np.zeros((len(atoms), max(lmax.values())+1), dtype=int)
    i = 0
    for iat, q in enumerate(atoms):
        for l in range(lmax[q]+1):
            idx[iat,l] = i
            i += (2*l+1)*nmax[(q,l)]
    return idx


def do_work_b(idx, nmax, conf, ref_elem, path_over, path_kern, Bmat):

  over = equistore.load(f"{path_over}{conf}.npz")
  k_NM = equistore.load(f"{path_kern}{conf}.dat.npz")

  for (l1, l2, q1, q2), oblock in over:
      msize1 = 2*l1+1;
      msize2 = 2*l2+1;
      nsize1 = nmax[(q1,l1)]
      nsize2 = nmax[(q2,l2)]
      kblock1 = k_NM.block(spherical_harmonics_l=l1, species_center=q1)
      kblock2 = k_NM.block(spherical_harmonics_l=l2, species_center=q2)
      oval = oblock.values.reshape(len(kblock1.samples), len(kblock2.samples), msize1, msize2, nsize1, nsize2)

      for iiref1, iref1 in enumerate(np.where(ref_elem==q1)[0]):
          for iiref2, iref2 in enumerate(np.where(ref_elem==q2)[0]):
              if iref1>iref2:
                  continue
              # dB = np.einsum('AMJ,AaMmNn,amj->NnJj', kblock1.values[:,:,:,iiref1], oval[:,:,:,:,:,:], kblock2.values[:,:,:,iiref2])
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


def get_b(lmax, nmax, totsize, ref_elem,
          nfrac, ntrains, trrange,
          path_over, path_kern, paths_bmat):

  ntrains = np.pad(ntrains, (0, 1), 'constant', constant_values=0)
  print_batches(sys.stdout, nfrac, ntrains, paths_bmat)

  Bmat = np.zeros(symsize(totsize))
  idx = sparseindices_fill(lmax, nmax, ref_elem)

  for ifrac in range(nfrac):
      for imol in range(ntrains[ifrac-1], ntrains[ifrac]):
        print(f'{nproc:4d}: {imol:4d}')
        do_work_b(idx, nmax, trrange[imol], ref_elem, path_over, path_kern, Bmat)
      print(Bmat)
      Bmat.tofile(paths_bmat[ifrac])
