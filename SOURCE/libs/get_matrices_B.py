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

def kernsparseindices_fill(
    nat,
    llmax ,
    M,
    atomcount,
    ref_elem,
    alnum):

  kernsparseindices = np.zeros((M, llmax+1, nat), dtype=int);
  i = 0;
  for iref in range(M):
    a = ref_elem[iref];
    al = alnum[a];
    for l in range(al):
      msize = 2*l+1;
      for iat in range(atomcount[a]):
        kernsparseindices[iref,l,iat] = i;
        i += msize*msize;
  return kernsparseindices;

def ao_fill(ref_elem, alnum, annum):
  aoref = []
  for iref, a in enumerate(ref_elem):
      for l in range(alnum[a]):
          msize = 2*l+1;
          for n in range(annum[a][l]):
              for im in range(msize):
                x = {'im'   : im,
                     'n'    : n,
                     'l'    : l,
                     'a'    : a,
                     'iref' : iref}
                aoref.append(x)
  return aoref;



def print_batches(f, nfrac, ntrains, paths):
    if nproc==0:
        for i in range(nfrac):
           print(f"batch {i:2d} [{ntrains[i-1]}--{ntrains[i]}):\t {paths[i]}", file=f);
        print(flush=True, file=f)
    if USE_MPI:
        MPI_Barrier(MPI_COMM_WORLD);


def sparseindices_fill(
    nat,
    llmax,
    nnmax,
    alnum,
    annum,
    atom_elem):

  sparseindices = np.zeros(((llmax+1) , nnmax , nat), dtype=int)
  i = 0;
  for iat in range(nat):
    a = atom_elem[iat];
    al = alnum[a];
    for l in range(al):
      msize = 2*l+1;
      anc   = annum[a][l];
      for n in range(anc):
        sparseindices[l,n,iat] = i;
        i += msize;
  return sparseindices;


def do_work_b(
    el_dict,
    totsize,
    nelem,
    llmax,
    nnmax,
    M,
    natmax,
    nat,
    conf,
    nao,
    kernsize,
    atomicindx,
    atomcount ,
    atom_elem ,
    ref_elem  ,
    alnum     ,
    annum     ,
    aoref     ,
    path_over,
    path_kern,
    Bmat):


  overlaps = np.load(f"{path_over}{conf}.npy")
  kernels  = np.loadtxt(f"{path_kern}{conf}.dat")
  over = equistore.load(f"{path_over}{conf}.npz")
  k_NM = equistore.load(f"{path_kern}{conf}.dat.npz")

  sparseindices = sparseindices_fill(nat, llmax, nnmax, alnum, annum, atom_elem);
  kernsparseindices = kernsparseindices_fill(nat, llmax , M, atomcount, ref_elem , alnum);

  sparseindices_bmat = sparseindices_fill(len(ref_elem), llmax, nnmax, alnum, annum, ref_elem);




  iiref1 = {q1: 0 for q1 in el_dict.values()}

  for iref1, a1 in enumerate(ref_elem):
      q1 = el_dict[a1]
      for l1 in range(alnum[a1]):
          msize1 = 2*l1+1;
          if (l1,q1) in k_NM.keys:
              kblock1 = k_NM.block(spherical_harmonics_l=l1, species_center=q1)
          for n1 in range(annum[a1][l1]):
              for im1 in range(msize1):
                i1 = sparseindices_bmat[l1,n1,iref1]+im1
                print(i1)

                iiref2 = {q2: 0 for q2 in el_dict.values()}

                i2 = 0
                for iref2, a2 in enumerate(ref_elem):
                    q2 = el_dict[a2]
                    for l2 in range(alnum[a2]):
                        msize2 = 2*l2+1;
                        if (l2,q2) in k_NM.keys:
                            kblock2 = k_NM.block(spherical_harmonics_l=l2, species_center=q2)
                        if (l1,q1) in k_NM.keys and (l2,q2) in k_NM.keys:
                            oblock = over.block(spherical_harmonics_l1=l1, species_center1=q1,
                                                spherical_harmonics_l2=l2, species_center2=q2)
                        for n2 in range(annum[a2][l2]):
                            for im2 in range(msize2):
                                msize2 = 2*l2+1;

                                i2 = sparseindices_bmat[l2,n2,iref2]+im2

                                dB = 0.0;
                                for icel1 in range(atomcount[a1]):
                                  iat = atomicindx[a1][icel1];
                                  sp1 = sparseindices    [l1,n1,iat];
                                  for icel2 in range(atomcount[a2]):
                                    k1  = kblock1.values[icel1,:,im1,iiref1[q1]]
                                    jat = atomicindx[a2][icel2];
                                    sp2 = sparseindices    [l2,n2,jat];
                                    o = overlaps[sp1:sp1+msize1, sp2:sp2+msize2]
                                    k2 = kblock2.values[icel2,:,im2,iiref2[q2]]
                                    dB += k1 @ o @ k2



                                if i1<=i2:
                                    Bmat[mpos(i1,i2)] += dB;
                                i2 += 1
                    iiref2[q2]+=1
      iiref1[q1]+=1


def get_b(
    el_dict,
    totsize,
    nelem  ,
    llmax  ,
    nnmax  ,
    M      ,
    ntrain ,
    natmax ,
    nfrac,
    ntrains   , #[nfrac],                  //  nfrac
    atomicindx, #[ntrain][nelem][natmax], // ntrain*nelem*natmax
    atomcount , #[ntrain][nelem],         // ntrain*nelem
    trrange   , #[ntrain],                // ntrain
    natoms    , #[ntrain],                // ntrain
    totalsizes, #[ntrain],                // ntrain
    kernsizes , #[ntrain],                // ntrain
    atom_elem , #[ntrain][natmax],        // ntrain*natmax
    ref_elem  , #[M],                     // M
    alnum     , #[nelem],                 // nelem
    annum     , #[nelem][llmax+1],        // nelem*(llmax+1)
    path_over,
    path_kern,
    paths_bmat):

  ntrains = np.pad(ntrains, (0, 1), 'constant', constant_values=0)
  print_batches(sys.stdout, nfrac, ntrains, paths_bmat)

  Bmat = np.zeros(symsize(totsize))
  aoref = ao_fill(ref_elem, alnum, annum);

  for ifrac in range(nfrac):
      for imol in range(ntrains[ifrac-1], ntrains[ifrac]):
        print(f'{nproc:4d}: {imol:4d}')
        do_work_b(
          el_dict,
          totsize,
          nelem  ,
          llmax  ,
          nnmax  ,
          M      ,
          natmax ,
          natoms    [imol],
          trrange   [imol],
          totalsizes[imol],
          kernsizes [imol],
          atomicindx[imol],
          atomcount [imol],
          atom_elem [imol],
          ref_elem  ,
          alnum     ,
          annum     ,
          aoref     ,
          path_over,
          path_kern,
          Bmat);
      print(Bmat)
      Bmat.tofile(paths_bmat[ifrac])
  return 0;





