import sys
import numpy as np
import equistore
from libs.tmap import vector2tmap, tmap2vector

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


#  static void print_mem(const int totsize, const int ntrain, FILE * f){
#
#    const double b2mib = 1.0/(1<<20);
#    const double b2gib = 1.0/(1<<30);
#    size_t size = (symsize(totsize))*sizeof(double);
#
#    fprintf(f, "\
#        Problem dimensionality = %d\n\
#        Number of training molecules = %d\n\
#        output: %16zu bytes (%10.2lf MiB, %6.2lf GiB)\n\n",
#        totsize, ntrain, size, size*b2mib, size*b2gib);
#    fflush(f);
#    return;
#  }
#
#  #ifdef USE_MPI
#  static void print_nodes(FILE * f){
#
#    char processor_name[MPI_MAX_PROCESSOR_NAME];
#    int name_len;
#    MPI_Get_processor_name(processor_name, &name_len);
#
#    char buf[256];
#    snprintf(buf, sizeof(buf), " proc %4d : %s\n", nproc, processor_name);
#
#    if(nproc){
#      MPI_Send(buf, strlen(buf)+1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
#    }
#    else{
#      fputs(buf, f);
#      MPI_Status status;
#      for(int i=1; i<Nproc; i++) {
#        MPI_Recv(buf, sizeof(buf), MPI_CHAR, i, 0, MPI_COMM_WORLD, &status);
#        fputs(buf, f);
#      }
#      fputs("\n", f);
#      fflush(f);
#    }
#    MPI_Barrier(MPI_COMM_WORLD);
#    return;
#  }

def print_batches(f, nfrac, ntrains, paths):
    if nproc==0:
        for i in range(nfrac):
           print(f"batch {i:2d} [{ntrains[i-1]}--{ntrains[i]}):\t {paths[i]}", file=f);
        print(flush=True, file=f)
    if USE_MPI:
        MPI_Barrier(MPI_COMM_WORLD);


#
#  static void send_jobs(const int bra, const int ket){
#    for(int imol=bra; imol<ket+(Nproc-1); imol++){
#      MPI_Status status;
#      int p[2];
#      MPI_Recv(p, 2, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
#      p[1] = (imol<ket) ? imol : -1;
#      MPI_Send(p, 2, MPI_INT, p[0], 0, MPI_COMM_WORLD);
#      printf("%4d: %4d\n", p[0], p[1]);
#    }
#    return;
#  }
#
#  static int receive_job(int imol){
#    int p[] = {nproc, imol};
#    MPI_Status status;
#    MPI_Send(p, 2, MPI_INT, 0, 0, MPI_COMM_WORLD);
#    MPI_Recv(p, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
#    return p[1];
#  }
#
#  #endif
#
#  static void vec_print(size_t n, double * v, const char * mode, const char * fname){
#
#    FILE * f = fopen(fname, mode);
#    if(!f){
#      fprintf(stderr, "cannot open file %s", fname);
#      GOTOHELL;
#    }
#    for(int i=0; i<n; i++){
#      fprintf(f, "% 18.15e\n", v[i]);
#    }
#    fprintf(f, "\n");
#    fclose(f);
#    return;
#  }
#
#  static void vec_write(size_t n, double * v, const char * mode, const char * fname){
#
#    FILE * f = fopen(fname, mode);
#    if(!f){
#      fprintf(stderr, "cannot open file %s", fname);
#      GOTOHELL;
#    }
#    if(fwrite(v, sizeof(double), n, f) != n){
#      fprintf(stderr, "cannot write to file %s line %d", fname);
#      GOTOHELL;
#    }
#    fclose(f);
#    return;
#  }
#
#  static double * npy_read(int n, const char * fname){
#
#    // read simple 1d / symmetric 2d arrays of doubles only
#    static const size_t header_size = 128;
#
#    double * v = calloc(n, sizeof(double));
#    FILE   * f = fopen(fname, "r");
#    if(!f){
#      fprintf(stderr, "cannot open file %s", fname);
#      GOTOHELL;
#    }
#    if( fseek(f, header_size, SEEK_SET) ||
#        fread(v, sizeof(double), n, f)!=n){
#      fprintf(stderr, "cannot read file %s line %d", fname);
#      GOTOHELL;
#    }
#    fclose(f);
#    return v;
#  }
#

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
