#include <math.h>
#ifdef USE_MPI
#include <mpi.h>
#include <string.h>
#endif
#include "mylib.h"

int Nproc;
int nproc;

#define SPARSEIND(L,N,IAT) ( ((L) * nnmax + (N)) * nat + (IAT) )
#define KSPARSEIND(IREF,L,M,ICSPE) ( (((IREF) * (llmax+1) + (L)) *(2*llmax+1) + (M)) * nat + (ICSPE) )

static void print_mem(
    const int totsize ,
    const int llmax   ,
    const int nnmax   ,
    const int M       ,
    const int ntrain  ,
    const int natmax  ,
    const unsigned int const totalsizes[ntrain],
    const unsigned int const kernsizes [ntrain],
    const size_t bufsize,
    FILE * f){

  const double b2mib = 1.0/(1<<20);
  const double b2gib = 1.0/(1<<30);

  size_t size1 = (symsize(totsize))*sizeof(double);
  size_t size2 = 0;
  size_t size3 = sizeof(int)*(llmax+1)*nnmax*natmax + sizeof(int)*M*(llmax+1)*(2*llmax+1)*natmax + sizeof(ao_t)*totsize;
  size_t size4 = sizeof(double) * bufsize;
  for(int imol=0; imol<ntrain; imol++){
    size_t t2 = totalsizes[imol]*totalsizes[imol] + kernsizes[imol];
    if(t2>size2) size2 = t2;
  }
  size2 *= sizeof(double);

  fprintf(f, "\nProblem dimensionality = %d\nNumber of training molecules = %d\n\n\
      output: %16zu bytes (%10.2lf MiB, %6.2lf GiB)\n\
      input : %16zu bytes (%10.2lf MiB, %6.2lf GiB)\n\
      inner : %16zu bytes (%10.2lf MiB, %6.2lf GiB)\n\
      buffer: %16zu bytes (%10.2lf MiB, %6.2lf GiB)\n\n",
      totsize, ntrain,
      size1, size1*b2mib, size1*b2gib,
      size2, size2*b2mib, size2*b2gib,
      size3, size3*b2mib, size3*b2gib,
      size4, size4*b2mib, size4*b2gib
      );
  fflush(f);
  return;
}

#ifdef USE_MPI
static void print_nodes(FILE * f){

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  char buf[256];
  snprintf(buf, sizeof(buf), " proc %4d : %s\n", nproc, processor_name);

  if(nproc){
    MPI_Send(buf, strlen(buf)+1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
  }
  else{
    fputs(buf, f);
    MPI_Status status;
    for(int i=1; i<Nproc; i++) {
      MPI_Recv(buf, sizeof(buf), MPI_CHAR, i, 0, MPI_COMM_WORLD, &status);
      fputs(buf, f);
    }
  }
  return;
}
#endif

static void vec_print(size_t n, double * v, const char * mode, const char * fname){

  FILE * f = fopen(fname, mode);
  if(!f){
    GOTOHELL;
  }
  for(int i=0; i<n; i++){
    fprintf(f, "% 18.15e\n", v[i]);
  }
  fprintf(f, "\n");
  fclose(f);
  return;
}

static void vec_write(size_t n, double * v, const char * mode, const char * fname){

  FILE * f = fopen(fname, mode);
  if(!f){
    GOTOHELL;
  }
  if(fwrite(v, sizeof(double), n, f) != n){
    GOTOHELL;
  }
  fclose(f);
  return;
}

static double * npy_read(int n, const char * fname){

  // read simple 1d / symmetric 2d arrays of doubles only
  static const size_t header_size = 128;

  FILE * f = fopen(fname, "r");
  if(!f){
    return NULL;
  }

  fseek(f, header_size, SEEK_SET);
  double * v = calloc(n, sizeof(double));
  int ret = fread(v, sizeof(double), n, f);
  fclose(f);

  if(ret!=n){
    free(v);
    return NULL;
  }

  return v;
}

static int * sparseindices_fill(
    const int nat,
    const int llmax,
    const int nnmax,
    const unsigned int const alnum[],           //  nelem
    const unsigned int const annum[][llmax+1],  //  nelem*(llmax+1)
    const unsigned int const atom_elem[]        // natmax
    ){

  int * sparseindices = calloc((llmax+1) * nnmax * nat, sizeof(int));
  int i = 0;
  for(int iat=0; iat<nat; iat++){
    int a = atom_elem[iat];
    int al = alnum[a];
    for(int l=0; l<al; l++){
      int msize = 2*l+1;
      int anc   = annum[a][l];
      for(int n=0; n<anc; n++){
        sparseindices[ SPARSEIND(l,n,iat)] = i;
        i += msize;
      }
    }
  }
  return sparseindices;
}

static int * kernsparseindices_fill(
    const int nat,
    const int llmax ,
    const int M,
    const unsigned int const atomcount[],  // nelem
    const unsigned int const ref_elem[M],
    const unsigned int const alnum[]       // nelem
    ){

  int * kernsparseindices = calloc(sizeof(int) *M *(llmax+1) *(2*llmax+1) * nat, 1);
  int i = 0;
  for(int iref=0; iref<M; iref++){
    int a = ref_elem[iref];
    int al = alnum[a];
    for(int l=0; l<al; l++){
      int msize = 2*l+1;
      for(int im=0; im<msize; im++){
        for(int iat=0; iat<atomcount[a]; iat++){
          kernsparseindices[KSPARSEIND(iref,l,im,iat)] = i;
          i += msize;
        }
      }
    }
  }
  return kernsparseindices;
}

static void do_work_a(
    const int from, const int to,
    const int totsize  ,
    const int nelem ,
    const int llmax    ,
    const int nnmax    ,
    const int M        ,
    const int natmax   ,
    const unsigned int const atomicindx[][nelem][natmax], //  ntrain*nelem*natmax
    const unsigned int const atomcount [][nelem],         //  ntrain*nelem
    const unsigned int const trrange   [],                //  ntrain
    const unsigned int const natoms    [],                //  ntrain
    const unsigned int const totalsizes[],                //  ntrain
    const unsigned int const kernsizes [],                //  ntrain
    const unsigned int const atom_elem [][natmax],        //  ntrain*natmax
    const unsigned int const ref_elem  [M],
    const unsigned int const alnum     [nelem],
    const unsigned int const annum     [nelem][llmax+1],
    const char * const path_proj,
    const char * const path_kern,
    double * Avec){

  if(to==from){
    return;
  }

  ao_t * aoref = ao_fill(totsize, llmax, M, ref_elem, alnum, annum);

  for(int imol=from; imol<to; imol++){

    printf("%4d: %4d\n", nproc, imol-from);

    const int nat = natoms[imol];
    const int nao = totalsizes[imol];

    const int conf = trrange[imol];
    char file_proj[512], file_kern[512];
    sprintf(file_proj, "%s%d.dat", path_proj, conf);
    sprintf(file_kern, "%s%d.dat", path_kern, conf);

    double * projections = vec_readtxt(nao, file_proj);
    double * kernels     = vec_readtxt(kernsizes[imol], file_kern);

    int * sparseindices = sparseindices_fill(nat, llmax, nnmax, alnum, annum, atom_elem[imol]);
    int * kernsparseindices = kernsparseindices_fill(nat, llmax , M, atomcount[imol], ref_elem , alnum);

#pragma omp parallel shared(Avec)
#pragma omp for schedule(dynamic)
    for(int i1=0; i1<totsize; i1++){
      int iref1 = aoref[i1].iref;
      int im1   = aoref[i1].im;
      int n1    = aoref[i1].n;
      int l1    = aoref[i1].l;
      int a1    = aoref[i1].a;
      int msize1 = 2*l1+1;
      double dA = 0.0;
      for(int icel1=0; icel1<atomcount[imol][a1]; icel1++){
        int iat = atomicindx[imol][a1][icel1];
        int sk1 = kernsparseindices[KSPARSEIND(iref1,l1,im1,icel1)];
        int sp1 = sparseindices    [SPARSEIND(l1,n1,iat)];
        for(int imm1=0; imm1<msize1; imm1++){
          dA += projections[sp1+imm1] * kernels[sk1+imm1];
        }
      }
      Avec[i1] += dA;
    }
    free(kernsparseindices);
    free(sparseindices);
    free(projections);
    free(kernels);
  }
  printf("%4d: finished work\n", nproc);

  free(aoref);
  return;
}

static void do_work_b(
    const int from, const int to,
    const int totsize,
    const int nelem  ,
    const int llmax  ,
    const int nnmax  ,
    const int M      ,
    const int natmax ,
    const unsigned int const atomicindx[][nelem][natmax], //  ntrain*nelem*natmax
    const unsigned int const atomcount [][nelem],         //  ntrain*nelem
    const unsigned int const trrange   [],                //  ntrain
    const unsigned int const natoms    [],                //  ntrain
    const unsigned int const totalsizes[],                //  ntrain
    const unsigned int const kernsizes [],                //  ntrain
    const unsigned int const atom_elem [][natmax],        //  ntrain*natmax
    const unsigned int const ref_elem  [M],
    const unsigned int const alnum     [nelem],
    const unsigned int const annum     [nelem][llmax+1],
    const char * const path_over,
    const char * const path_kern,
    double * Bmat){

  if(to==from){
    return;
  }

  ao_t * aoref = malloc(sizeof(ao_t)*totsize);

  int i = 0;
  for(int iref=0; iref<M; iref++){
    int a = ref_elem[iref];
    int al = alnum[a];
    for(int l=0; l<al; l++){
      int msize = 2*l+1;
      int anc   = annum[a][l];
      for(int n=0; n<anc; n++){
        for(int im=0; im<msize; im++){
          aoref[i].im   = im;
          aoref[i].n    = n;
          aoref[i].l    = l;
          aoref[i].a    = a;
          aoref[i].iref = iref;
          i++;
        }
      }
    }
  }

  for(int imol=from; imol<to; imol++){

    printf("%4d: %4d\n", nproc, imol-from);

    const int nat = natoms[imol];
    const int nao = totalsizes[imol];

    const int conf = trrange[imol];
    char file_over[512], file_kern[512];
    sprintf(file_over, "%s%d.npy", path_over, conf);
    sprintf(file_kern, "%s%d.dat", path_kern, conf);

    double * overlaps = npy_read(nao*nao, file_over);
    double * kernels  = vec_readtxt(kernsizes[imol], file_kern);

    int * sparseindices = sparseindices_fill(nat, llmax, nnmax, alnum, annum, atom_elem[imol]);
    int * kernsparseindices = kernsparseindices_fill(nat, llmax , M, atomcount[imol], ref_elem , alnum);

#pragma omp parallel shared(Bmat)
#pragma omp for schedule(dynamic)
    for(int i1=0; i1<totsize; i1++){
      int iref1 = aoref[i1].iref;
      int im1   = aoref[i1].im;
      int n1    = aoref[i1].n;
      int l1    = aoref[i1].l;
      int a1    = aoref[i1].a;
      int msize1 = 2*l1+1;

      for(int i2=i1; i2<totsize; i2++){
        int iref2 = aoref[i2].iref;
        int im2   = aoref[i2].im;
        int n2    = aoref[i2].n;
        int l2    = aoref[i2].l;
        int a2    = aoref[i2].a;
        int msize2 = 2*l2+1;
        double dB = 0.0;
        for(int icel1=0; icel1<atomcount[imol][a1]; icel1++){
          int iat = atomicindx[imol][a1][icel1];
          int sk1 = kernsparseindices[KSPARSEIND(iref1, l1, im1, icel1)];
          int sp1 = sparseindices    [SPARSEIND(l1,n1,iat)];
          for(int imm1=0; imm1<msize1; imm1++){
            double Btemp = 0.0;
            for(int icel2=0; icel2<atomcount[imol][a2]; icel2++){
              int jat = atomicindx[imol][a2][icel2];
              int sk2 = kernsparseindices[KSPARSEIND(iref2, l2, im2, icel2)];
              int sp2 = sparseindices    [SPARSEIND(l2,n2,jat)];
              for(int imm2=0; imm2<msize2; imm2++){
                Btemp += overlaps[(sp2+imm2)*nao+(sp1+imm1)] * kernels[sk2+imm2];
              }
            }
            dB += Btemp * kernels[sk1+imm1];
          }
        }
        Bmat[mpos(i1,i2)] += dB;
      }
    }
    free(kernsparseindices);
    free(sparseindices);
    free(overlaps);
    free(kernels);
  }
  printf("%4d: finished work\n", nproc);

  free(aoref);
  return;
}

int get_a(
    const int totsize  ,
    const int nelem ,
    const int llmax    ,
    const int nnmax    ,
    const int M        ,
    const int ntrain   ,
    const int natmax   ,
    const unsigned int const atomicindx[ntrain][nelem][natmax],  //  ntrain*nelem*natmax
    const unsigned int const atomcount [ntrain][nelem],  //  ntrain*nelem
    const unsigned int const trrange   [ntrain],  //  ntrain
    const unsigned int const natoms    [ntrain],  //  ntrain
    const unsigned int const totalsizes[ntrain],  //  ntrain
    const unsigned int const kernsizes [ntrain],  //  ntrain
    const unsigned int const atom_elem   [ntrain][natmax]   ,  //  ntrain*natmax
    const unsigned int const ref_elem [M],  //  M
    const unsigned int const alnum     [nelem]     ,  //  nelem
    const unsigned int const annum     [nelem][llmax+1],  //  nelem*(llmax+1)
    const char * const path_proj,
    const char * const path_kern,
    const char * const path_avec
    ){

#ifdef USE_MPI
  int argc = 1;
  char * argv0 = "get_matrices";
  char ** argv = &argv0;
  MPI_Init (&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD, &Nproc);
  MPI_Comm_rank (MPI_COMM_WORLD, &nproc);
  print_nodes(stdout);
#else
  nproc = 0;
  Nproc = 1;
#endif

#ifdef USE_MPI
  double t = 0.0;
  if(nproc == 0){
    t = MPI_Wtime ();
  }
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  double * Avec = calloc(sizeof(double)*totsize, 1);

  int start[Nproc+1];
  int div = ntrain/Nproc;
  int rem = ntrain%Nproc;
  start[0] = 0;
  for(int i=0; i<Nproc; i++){
    start[i+1] = start[i] + ( (i<rem) ? (div+1) : div );
  }

  do_work_a(start[nproc], start[nproc+1],
      totsize ,
      nelem,
      llmax   ,
      nnmax   ,
      M       ,
      natmax  ,
      atomicindx,
      atomcount ,
      trrange   ,
      natoms    ,
      totalsizes,
      kernsizes ,
      atom_elem   ,
      ref_elem ,
      alnum     ,
      annum     ,
      path_proj,
      path_kern,
      Avec);

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  if(!nproc){
    t = MPI_Wtime () - t;
    fprintf(stderr, "t=%4.2lf\n", t);
  }
#endif

  if(Nproc==1){
    vec_print(totsize, Avec, "w", path_avec);
  }
#ifdef USE_MPI
  else{
    double * AVEC = NULL;
    if(!nproc){
      AVEC = calloc(sizeof(double)*totsize, 1);
    }

    MPI_Reduce (Avec, nproc?NULL:AVEC, totsize, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(!nproc){
      vec_print(totsize, AVEC, "w", path_avec);
    }

    free(AVEC);
  }
#endif
  free(Avec);

#ifdef USE_MPI
  MPI_Finalize ();
#endif
  return 0;
}

int get_b(
    const int totsize,
    const int nelem  ,
    const int llmax  ,
    const int nnmax  ,
    const int M      ,
    const int ntrain ,
    const int natmax ,
    const unsigned int const atomicindx[ntrain][nelem][natmax], // ntrain*nelem*natmax
    const unsigned int const atomcount [ntrain][nelem],         // ntrain*nelem
    const unsigned int const trrange   [ntrain],                // ntrain
    const unsigned int const natoms    [ntrain],                // ntrain
    const unsigned int const totalsizes[ntrain],                // ntrain
    const unsigned int const kernsizes [ntrain],                // ntrain
    const unsigned int const atom_elem [ntrain][natmax],        // ntrain*natmax
    const unsigned int const ref_elem  [M],                     // M
    const unsigned int const alnum     [nelem],                 // nelem
    const unsigned int const annum     [nelem][llmax+1],        // nelem*(llmax+1)
    const char * const path_over,
    const char * const path_kern,
    const char * const path_bmat
    ){

#ifdef USE_MPI
  int argc = 1;
  char * argv0 = "get_matrices";
  char ** argv = &argv0;
  MPI_Init (&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD, &Nproc);
  MPI_Comm_rank (MPI_COMM_WORLD, &nproc);
  print_nodes(stdout);
#else
  nproc = 0;
  Nproc = 1;
#endif

#ifdef USE_MPI
  size_t bufsize = (1<<30)/sizeof(double);  // number of doubles to take 1 GiB
  if(bufsize > symsize(totsize)){
    bufsize = symsize(totsize);
  }
  if(Nproc == 1){
    bufsize = 0;
  }

  double t = 0.0;
  if(nproc == 0){
    print_mem(totsize, llmax, nnmax, M, ntrain, natmax, totalsizes, kernsizes, bufsize, stdout);
    t = MPI_Wtime ();
  }
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  double * Bmat = calloc(sizeof(double)*symsize(totsize), 1);

  int start[Nproc+1];
  int div = ntrain/Nproc;
  int rem = ntrain%Nproc;
  start[0] = 0;
  for(int i=0; i<Nproc; i++){
    start[i+1] = start[i] + ( (i<rem) ? (div+1) : div );
  }

  do_work_b(start[nproc], start[nproc+1],
      totsize,
      nelem  ,
      llmax  ,
      nnmax  ,
      M      ,
      natmax ,
      atomicindx,
      atomcount ,
      trrange   ,
      natoms    ,
      totalsizes,
      kernsizes ,
      atom_elem ,
      ref_elem  ,
      alnum     ,
      annum     ,
      path_over,
      path_kern,
      Bmat);

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  if(!nproc){
    t = MPI_Wtime () - t;
    fprintf(stderr, "t=%4.2lf\n", t);
  }
#endif

  if(Nproc==1){
    vec_write(symsize(totsize), Bmat, "w", path_bmat);
  }
#ifdef USE_MPI
  else{
    double * BMAT = NULL;
    if(!nproc){
      BMAT = calloc(sizeof(double)*bufsize, 1);
    }

    size_t div = symsize(totsize)/bufsize;
    size_t rem = symsize(totsize)%bufsize;
    for(size_t i=0; i<=div; i++){
      size_t size = i<div?bufsize:rem;
      if(!size){
        break;
      }
      MPI_Reduce (Bmat+i*bufsize, nproc?NULL:BMAT, size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      if(!nproc){
        vec_write(size, BMAT, i?"a":"w", path_bmat);
        printf("chunk #%d/%d written\n", i+1, rem?(div+1):div);
      }
    }

    free(BMAT);
  }
#endif
  free(Bmat);

#ifdef USE_MPI
  MPI_Finalize ();
#endif
  return 0;
}

