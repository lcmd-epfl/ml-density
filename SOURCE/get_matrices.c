#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#ifdef USE_MPI
#include <mpi.h>
#include <string.h>
#endif

int Nproc;
int nproc;

#define printalive printf("alive @ %s:%d\n", __FILE__, __LINE__)
#define GOTOHELL { \
  fprintf(stderr, "%s:%d %s() -- ", \
      __FILE__, __LINE__, __FUNCTION__); \
  abort(); }

#define SPARSEIND(L,N,IAT) ( ((L) * nnmax + (N)) * nat + (IAT) )
#define KSPARSEIND(IREF,L,M,ICSPE) ( (((IREF) * (llmax+1) + (L)) *(2*llmax+1) + (M)) * nat + (ICSPE) )

static inline size_t symsize(size_t M){
  return (M*(M+1))>>1;
}

static inline size_t mpos(size_t i, size_t j){
  /* A[i+j*(j+1)/2], i <= j, 0 <= j < N */
  return (i)+(((j)*((j)+1))>>1);
}
#define MPOSIF(i,j)   ((i)<=(j)? mpos((i),(j)):mpos((j),(i)))

typedef struct {
  int im;
  int n;
  int l;
  int a;
  int iref;
} ao_t;

static void print_mem(
    const int totsize  ,
    const int llmax    ,
    const int nnmax    ,
    const int M        ,
    const int ntrain   ,
    const int natmax   ,
    const int * const totalsizes,
    const int * const kernsizes,
    const size_t bufsize,
    FILE * f){

  const double b2mib = 1.0/(1<<20);
  const double b2gib = 1.0/(1<<30);

  size_t size1 = (2*totsize+symsize(totsize))*sizeof(double);
  size_t size2 = 0;
  size_t size3 = sizeof(int)*(llmax+1)*nnmax*natmax + sizeof(int)*M*(llmax+1)*(2*llmax+1)*natmax + sizeof(ao_t)*totsize;
  size_t size4 = sizeof(double) * bufsize;
  for(int itrain=0; itrain<ntrain; itrain++){
    size_t t2 = totalsizes[itrain]*(totalsizes[itrain]+1)+kernsizes[itrain];
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

static void vec_print(int n, double * v, const char * fname){

  FILE * f = fopen(fname, "w");
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

static void mx_nosym_print(size_t n, double * a, const char * fname){

  FILE * f = fopen(fname, "w");
  if(!f){
    GOTOHELL;
  }
  for(size_t i=0; i<n; i++){
    for(size_t j=0; j<=i; j++){
      fprintf(f, "% 18.15e   ", a[mpos(j,i)]);
    }
    for(size_t j=i+1; j<n; j++){
      fprintf(f, "% 18.15e   ", a[mpos(i,j)]);
    }
    fprintf(f, "\n");
  }
  fprintf(f, "\n");
  fclose(f);
  return;
}

static double * vec_read(int n, char * fname){

  double * v = malloc(sizeof(double)*n);
  FILE   * f = fopen(fname, "r");

  for(int i=0; i<n; i++){
    if(fscanf(f, "%lf", v+i)!=1){
      GOTOHELL;
    }
  }
  fclose(f);
  return v;
}

static void do_work(
    const int from, const int to,
    const int totsize  ,
    const int nspecies ,
    const int llmax    ,
    const int nnmax    ,
    const int M        ,
    const int natmax   ,
    const int * const atomicindx,
    const int * const atomcount ,
    const int * const trrange   ,
    const int * const natoms    ,
    const int * const totalsizes,
    const int * const kernsizes ,
    const int * const atomspe   ,
    const int * const specarray ,
    const int * const almax     ,
    const int * const ancut     ,
    const char * const path_proj,
    const char * const path_over,
    const char * const path_kern,
    double * Avec, double * Bmat){

  if(to==from){
    return;
  }

  ao_t * ao = malloc(sizeof(ao_t)*totsize);

  int i = 0;
  for(int iref=0; iref<M; iref++){
    int a = specarray[iref];
    int al = almax[a];
    for(int l=0; l<al; l++){
      int msize = 2*l+1;
      int anc   = ancut[ a * (llmax+1) + l ];
      for(int n=0; n<anc; n++){
        for(int im=0; im<msize; im++){
          ao[i].im = im;
          ao[i].n  = n;
          ao[i].l  = l;
          ao[i].a  = a;
          ao[i].iref = iref;
          i++;
        }
      }
    }
  }

  for(int itrain=from; itrain<to; itrain++){

    printf("%4d %4d\n", nproc, itrain-from);

    const int nat = natoms[itrain];
    const int nao = totalsizes[itrain];

    const int conf = trrange[itrain];
    char file_proj[512], file_over[512], file_kern[512];
    sprintf(file_proj, "%s%d.dat", path_proj, conf);
    sprintf(file_over, "%s%d.dat", path_over, conf);
    sprintf(file_kern, "%s%d.dat", path_kern, conf);

    double * projections = vec_read(nao,     file_proj);
    double * overlaps    = vec_read(nao*nao, file_over);
    double * kernels     = vec_read( kernsizes[itrain], file_kern);

    int * sparseindexes = calloc(sizeof(int) * (llmax+1) * nnmax * nat, 1);
    int * kernsparseindexes = calloc(sizeof(int) *M *(llmax+1) *(2*llmax+1) * nat, 1);

    int it = 0;
    for(int iat=0; iat<nat; iat++){
      int a = atomspe[itrain*natmax+iat];
      int al = almax[a];
      for(int l=0; l<al; l++){
        int msize = 2*l+1;
        int anc   = ancut[ a * (llmax+1) + l ];
        for(int n=0; n<anc; n++){
          sparseindexes[ SPARSEIND(l,n,iat)] = it;
          it += msize;
        }
      }
    }

    int ik = 0;
    for(int iref=0; iref<M; iref++){
      int a = specarray[iref];
      int al = almax[a];
      for(int l=0; l<al; l++){
        int msize = 2*l+1;
        for(int im=0; im<msize; im++){
          for(int iat=0; iat<atomcount[ itrain*nspecies+a ]; iat++){
            kernsparseindexes[KSPARSEIND(iref,l,im,iat)] = ik;
            ik += msize;
          }
        }
      }
    }

    for(int i1=0; i1<totsize; i1++){
      int iref1 = ao[i1].iref;
      int  im1 = ao[i1].im;
      int  n1  = ao[i1].n;
      int  l1  = ao[i1].l;
      int  a1  = ao[i1].a;
      int msize1 = 2*l1+1;
      double dA = 0.0;
      for(int icspe1=0; icspe1<atomcount[itrain*nspecies+a1]; icspe1++){
        int iat = atomicindx[ itrain*nspecies*natmax+a1*natmax+icspe1 ];
        int sk1 = kernsparseindexes[KSPARSEIND(iref1,l1,im1,icspe1)];
        int sp1 = sparseindexes    [SPARSEIND(l1,n1,iat)];
        for(int imm1=0; imm1<msize1; imm1++){
          dA += projections[sp1+imm1] * kernels[sk1+imm1];
        }
      }
      Avec[i1] += dA;

      for(int i2=i1; i2<totsize; i2++){
        int iref2 = ao[i2].iref;
        int  im2 = ao[i2].im;
        int  n2  = ao[i2].n;
        int  l2  = ao[i2].l;
        int  a2  = ao[i2].a;
        int msize2 = 2*l2+1;
        double dB = 0.0;
        for(int icspe1=0; icspe1<atomcount[itrain*nspecies+a1]; icspe1++){
        int iat = atomicindx[ itrain*nspecies*natmax+a1*natmax+icspe1 ];
          int sk1 = kernsparseindexes[KSPARSEIND(iref1, l1, im1, icspe1)];
          int sp1 = sparseindexes    [SPARSEIND(l1,n1,iat)];
          for(int imm1=0; imm1<msize1; imm1++){
            double Btemp = 0.0;
            for(int icspe2=0; icspe2<atomcount[itrain*nspecies+a2]; icspe2++){
              int jat = atomicindx[ itrain*nspecies*natmax+a2*natmax+icspe2 ];
              int sk2 = kernsparseindexes[KSPARSEIND(iref2, l2, im2, icspe2)];
              int sp2 = sparseindexes    [SPARSEIND(l2,n2,jat)];
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
    free(kernsparseindexes);
    free(sparseindexes);
    free(projections);
    free(overlaps);
    free(kernels);
  }
  free(ao);
  return;
}


int get_matrices(
    const int totsize  ,
    const int nspecies ,
    const int llmax    ,
    const int nnmax    ,
    const int M        ,
    const int ntrain   ,
    const int natmax   ,
    const int * const atomicindx,  //  ntrain*nspecies*natmax
    const int * const atomcount ,  //  ntrain*nspecies
    const int * const trrange   ,  //  ntrain
    const int * const natoms    ,  //  ntrain
    const int * const totalsizes,  //  ntrain
    const int * const kernsizes ,  //  ntrain
    const int * const atomspe   ,  //  ntrain*natmax
    const int * const specarray ,  //  M
    const int * const almax     ,  //  nspecies
    const int * const ancut     ,  //  nspecies*(llmax+1)
    const char * const path_proj,
    const char * const path_over,
    const char * const path_kern,
    const char * const path_avec,
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

  double * Avec = calloc(sizeof(double)*totsize, 1);
  double * Bmat = NULL;
  if( (Nproc==1) || nproc){
    Bmat = calloc(sizeof(double)*symsize(totsize), 1);
  }
  double * AVEC = NULL;
  double * BMAT = NULL;
  if(!nproc){
    if(Nproc > 1){
      AVEC = calloc(sizeof(double)*totsize, 1);
      BMAT = calloc(sizeof(double)*symsize(totsize), 1);
    }
    else{
      AVEC = Avec;
      BMAT = Bmat;
    }
  }

  int start[Nproc+1];
  if(Nproc>1){
    int div = ntrain/(Nproc-1);
    int rem = ntrain%(Nproc-1);
    start[0] = 0;
    start[1] = 0;
    for(int i=1; i<Nproc; i++){
      start[i+1] = start[i] + ( ((i-1)<rem) ? (div+1) : div );
    }
  }
  else{
    start[0] = 0;
    start[1] = ntrain;
  }

  do_work(start[nproc], start[nproc+1],
      totsize ,
      nspecies,
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
      atomspe   ,
      specarray ,
      almax     ,
      ancut     ,
      path_proj,
      path_over,
      path_kern,
      Avec, Bmat);

#ifdef USE_MPI

  if(Nproc>1){

    MPI_Reduce (Avec, nproc?NULL:AVEC, totsize, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    // the 0th process did not work, but it has to send data
    double * buf = NULL;
    if(!nproc){
      buf = calloc(bufsize*sizeof(double), 1);
    }

    size_t div = symsize(totsize)/bufsize;
    size_t rem = symsize(totsize)%bufsize;
    for(size_t i=0; i<=div; i++){
      MPI_Reduce ( nproc ? (Bmat+i*bufsize):buf, nproc?NULL:(BMAT+i*bufsize),   i<div?bufsize:rem, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    free(buf);
  }

#endif

  if(nproc == 0){
#ifdef USE_MPI
    t = MPI_Wtime () - t;
    fprintf(stderr, "t=%4.2lf\n", t);
#endif
    vec_print(totsize, AVEC, path_avec);
    mx_nosym_print(totsize, BMAT, path_bmat);
  }

  free(Avec);
  free(Bmat);
  if(Nproc > 1){
    free(AVEC);
    free(BMAT);
  }

#ifdef USE_MPI
  MPI_Finalize ();
#endif
  return 0;
}

