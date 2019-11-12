#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#ifdef USE_MPI
#include <mpi.h>
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

static void vec_print(int n, double * v, const char * fname){

  FILE * f = fopen(fname, "w");
  for(int i=0; i<n; i++){
    fprintf(f, "% 18.15e\n", v[i]);
  }
  fprintf(f, "\n");
  fclose(f);
  return;
}

static void mx_print(int n, double * a, const char * fname){

  FILE * f = fopen(fname, "w");
  for(int i=0; i<n; i++){
    for(int j=0; j<n; j++){
      fprintf(f, "% 21.18e   ", a[i*n+j]);
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
    const int ntrain   ,
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

  for(int itrain=from; itrain<to; itrain++){

    const int nat = natoms[itrain];
    const int nao = totalsizes[itrain];

    //   ! read projections, overlaps and kernels for each training molecule
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


    int it1 = 0;
    for(int iat=0; iat<nat; iat++){
      int a1 = atomspe[itrain*natmax+iat];
      int al1 = almax[a1];
      for(int l1=0; l1<al1; l1++){
        int msize1 = 2*l1+1;
        int anc1   = ancut[ a1 * (llmax+1) + l1 ];
        for(int n1=0; n1<anc1; n1++){
          sparseindexes[ SPARSEIND(l1,n1,iat)] = it1;
          it1 += msize1;
        }
      }
    }

    int ik1 = 0;
    for(int iref1=0; iref1<M; iref1++){
      int a1 = specarray[iref1];
      int al1 = almax[a1];

      for(int l1=0; l1<al1; l1++){
        int msize1 = 2*l1+1;
        for(int im1=0; im1<msize1; im1++){
          for(int iat=0; iat<atomcount[ itrain*nspecies+a1 ]; iat++){
            kernsparseindexes[KSPARSEIND(iref1,l1,im1,iat)] = ik1;
            ik1 += msize1;
          }
        }
      }
    }

    //   ! Loop over 1st dimension
    int i1 = 0;
    for(int iref1=0; iref1<M; iref1++){
      int a1 = specarray[iref1];
      int al1 = almax[a1];
      for(int l1=0; l1<al1; l1++){
        int msize1 = 2*l1+1;
        int anc1   = ancut[ a1 * (llmax+1) + l1 ];
        for(int n1=0; n1<anc1; n1++){
          for(int im1=0; im1<msize1; im1++){
            // ! Collect contributions for 1st dimension
            for(int icspe1=0; icspe1<atomcount[itrain*nspecies+a1]; icspe1++){
              int iat = atomicindx[icspe1*nspecies*ntrain + a1*ntrain + itrain];
              int sk1 = kernsparseindexes[KSPARSEIND(iref1,l1,im1,icspe1)];
              int sp1 = sparseindexes    [SPARSEIND(l1,n1,iat)];
              for(int imm1=0; imm1<msize1; imm1++){
                Avec[i1] += projections[sp1+imm1] * kernels[sk1+imm1];
              }
            }

            int i2 = 0;
            for(int iref2=0; iref2<=iref1; iref2++){
              int a2 = specarray[iref2];
              int al2 = almax[a2];
              for(int l2=0; l2<al2; l2++){
                int msize2 = 2*l2+1;
                int anc2   = ancut[ a2 * (llmax+1) + l2 ];
                for(int n2=0; n2<anc2; n2++){
                  for(int im2=0; im2<msize2; im2++){
                    double contrB = 0.0;
                    for(int icspe1=0; icspe1<atomcount[itrain*nspecies+a1]; icspe1++){
                      int iat = atomicindx[icspe1*nspecies*ntrain + a1*ntrain + itrain];
                      int sk1 = kernsparseindexes[KSPARSEIND(iref1, l1, im1, icspe1)];
                      int sp1 = sparseindexes    [SPARSEIND(l1,n1,iat)];
                      for(int imm1=0; imm1<msize1; imm1++){
                        double Btemp = 0.0;
                        for(int icspe2=0; icspe2<atomcount[itrain*nspecies+a2]; icspe2++){
                          int jat = atomicindx[icspe2*nspecies*ntrain + a2*ntrain + itrain];
                          int sk2 = kernsparseindexes[KSPARSEIND(iref2, l2, im2, icspe2)];
                          int sp2 = sparseindexes    [SPARSEIND(l2,n2,jat)];
                          for(int imm2=0; imm2<msize2; imm2++){
                            Btemp += overlaps[(sp2+imm2)*nao+(sp1+imm1)] * kernels[sk2+imm2];
                          }
                        }
                        contrB += Btemp * kernels[sk1+imm1];
                      }
                    }
                    Bmat[i1*totsize+i2] += contrB;
                    if(iref2!=iref1){
                      Bmat[i2*totsize+i1] += contrB;
                    }
                    i2++;
                  }
                }
              }
            }
            i1++;
          }
        }
      }
    }
    free(kernsparseindexes);
    free(sparseindexes);
    free(projections);
    free(overlaps);
    free(kernels);
  }
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
#else
  nproc = 0;
  Nproc = 1;
#endif
  if (nproc == 0) {
    size_t size1 = totsize*(totsize+1)*sizeof(double);
    size_t size2 = 0;
    size_t size3 = 0;
    for(int itrain=0; itrain<ntrain; itrain++){
      size_t t2 = totalsizes[itrain]*(totalsizes[itrain]+1)+kernsizes[itrain];
      if(t2>size2) size2 = t2;
      size_t t3 = natoms[itrain];
      if(t3>size3) size3 = t3;
    }
    size2 *= sizeof(double);
    size3 *= sizeof(int) * (llmax+1)*(nnmax+(2*llmax+1)* M);

    fprintf(stderr, "Problem dimensionality = %d\nNumber of training molecules = %d\n\n", totsize, ntrain);
    fprintf(stderr, "\
        output: %16zu bytes (%10.2lf MiB, %6.2lf GiB)\n\
        input : %16zu bytes (%10.2lf MiB, %6.2lf GiB)\n\
        inner : %16zu bytes (%10.2lf MiB, %6.2lf GiB)\n",
        size1, size1/1048576.0, size1/1048576.0/1048576.0,
        size2, size2/1048576.0, size2/1048576.0/1048576.0,
        size3, size3/1048576.0, size3/1048576.0/1048576.0
        );
  }

  double t;
#ifdef USE_MPI
  if (nproc == 0) {
    t = MPI_Wtime ();
  }
#endif

#ifdef USE_MPI
  double * AVEC = calloc(sizeof(double)*totsize, 1);
  double * BMAT = calloc(sizeof(double)*totsize*totsize, 1);
#endif
  double * Avec = calloc(sizeof(double)*totsize, 1);
  double * Bmat = calloc(sizeof(double)*totsize*totsize, 1);

  int div = ntrain/Nproc;
  int rem = ntrain%Nproc;
  int start[Nproc+1];
  start[0] = 0;
  for(int i=0; i<Nproc; i++){
    start[i+1] = start[i] + ((i<rem)?div+1:div);
  }

  do_work(start[nproc], start[nproc+1],
      totsize ,
      nspecies,
      llmax   ,
      nnmax   ,
      M       ,
      ntrain  ,
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
  MPI_Reduce (Avec, AVEC, totsize,         MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce (Bmat, BMAT, totsize*totsize, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif

#ifdef USE_MPI
  if(nproc == 0){
    t = MPI_Wtime () - t;
    fprintf(stderr, "t=%4.2lf\n", t);
    vec_print(totsize, AVEC, path_avec);
    mx_print(totsize, BMAT, path_bmat);
  }
#else
    vec_print(totsize, Avec, path_avec);
    mx_print(totsize, Bmat, path_bmat);
#endif

  free(Avec);
  free(Bmat);

#ifdef USE_MPI
  free(AVEC);
  free(BMAT);
  MPI_Finalize ();
#endif
  return 0;
}

