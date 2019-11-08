#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

#define printalive printf("alive @ %s:%d\n", __FILE__, __LINE__)
#define GOTOHELL { \
  fprintf(stderr, "%s:%d %s() -- ", \
      __FILE__, __LINE__, __FUNCTION__); \
  abort(); }

int Nproc;
int nproc;

int llmax    =    5;
int nnmax    =   10;
int nspecies =    4;
int ntrain   =    8;
int M        =   32;
int natmax   =   24;
int totsize  = 2677;

const char path_proj[] = "BASELINED_PROJECTIONS/projections_conf";
const char path_over[] = "OVER_DAT/overlap_conf";
const char path_kern[] = "KERNELS/kernel_conf";

static inline int sparseind(int l, int n, int iat, int natoms) {
  return (l * nnmax + n) * natoms + iat;
}

static inline int ksparseind(int iref, int l, int m, int icspe, int natoms) {
  return ((iref * (llmax+1) + l) *(2*llmax+1) + m) * natoms + icspe;
}

void vec_print(int n, double * v, const char * s, FILE * f){
  for(int i=0; i<n; i++){
    fprintf(f, "% 20.15lf %s", v[i], s);
  }
  fprintf(f, "\n");
  return;
}

int * ivec_read(int n, char * fname){

  int  * v = malloc(sizeof(int)*n);
  FILE * f = fopen(fname, "r");

  for(int i=0; i<n; i++){
    if(fscanf(f, "%d", v+i)!=1){
      GOTOHELL;
    }
  }
  fclose(f);
  return v;
}

double * vec_read(int n, char * fname){

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


void do_work(
    int from, int to,
    int * atomicindx,
    int * atomcount ,
    int * trrange   ,
    int * natoms    ,
    int * totalsizes,
    int * kernsizes ,
    int * atomspe   ,
    int * specarray ,
    int * almax     ,
    int * ancut     ,
    double * Avec, double * Bmat){

  for(int itrain=from; itrain<to; itrain++){

    int nat = natoms[itrain];
    int nao = totalsizes[itrain];

    //   ! read projections, overlaps and kernels for each training molecule
    int conf = trrange[itrain];
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
          sparseindexes[ sparseind( l1,n1,iat, nat)] = it1;
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
            kernsparseindexes[ksparseind(iref1,l1,im1,iat, nat)] = ik1;
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
              int sk1 = kernsparseindexes[ ksparseind(iref1,l1,im1,icspe1, nat)];
              int sp1 = sparseindexes    [ sparseind (l1,n1,iat, nat)];
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
                      int sk1 = kernsparseindexes[ ksparseind( iref1, l1, im1, icspe1, nat)];
                      int sp1 = sparseindexes    [ sparseind (l1,n1,iat, nat)];
                      for(int imm1=0; imm1<msize1; imm1++){
                        double Btemp = 0.0;
                        for(int icspe2=0; icspe2<atomcount[itrain*nspecies+a2]; icspe2++){
                          int jat = atomicindx[icspe2*nspecies*ntrain + a2*ntrain + itrain];
                          int sk2 = kernsparseindexes[ ksparseind( iref2, l2, im2, icspe2, nat)];
                          int sp2 = sparseindexes    [ sparseind (l2,n2,jat, nat)];
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


int main(int argc, char ** argv){

  MPI_Init (&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD, &Nproc);
  MPI_Comm_rank (MPI_COMM_WORLD, &nproc);

  int * atomicindx;
  int * atomcount ;
  int * trrange   ;
  int * natoms    ;
  int * totalsizes;
  int * kernsizes ;
  int * atomspe   ;
  int * specarray ;
  int * almax     ;
  int * ancut     ;
  if(nproc == 0){
    atomicindx = ivec_read(natmax*nspecies*ntrain  ,  "atomicindx_training.dat"    );
    atomcount  = ivec_read(ntrain*nspecies         ,  "atom_counting_training.dat" );
    trrange    = ivec_read(ntrain                  ,  "train_configs.dat"          );
    natoms     = ivec_read(ntrain                  ,  "natoms_train.dat"           );
    totalsizes = ivec_read(ntrain                  ,  "total_sizes.dat"            );
    kernsizes  = ivec_read(ntrain                  ,  "kernel_sizes.dat"           );
    atomspe    = ivec_read(ntrain*natmax           ,  "atomic_species.dat"         );
    specarray  = ivec_read(M                       ,  "fps_species.dat"            );
    almax      = ivec_read(nspecies                ,  "almax.dat"                  );
    ancut      = ivec_read(nspecies*(llmax+1)      ,  "anmax.dat"                  );
  }
  else{
    atomicindx = calloc(sizeof(int)*natmax*nspecies*ntrain  ,  1 );
    atomcount  = calloc(sizeof(int)*ntrain*nspecies         ,  1 );
    trrange    = calloc(sizeof(int)*ntrain                  ,  1 );
    natoms     = calloc(sizeof(int)*ntrain                  ,  1 );
    totalsizes = calloc(sizeof(int)*ntrain                  ,  1 );
    kernsizes  = calloc(sizeof(int)*ntrain                  ,  1 );
    atomspe    = calloc(sizeof(int)*ntrain*natmax           ,  1 );
    specarray  = calloc(sizeof(int)*M                       ,  1 );
    almax      = calloc(sizeof(int)*nspecies                ,  1 );
    ancut      = calloc(sizeof(int)*nspecies*(llmax+1)      ,  1 );
  }
  MPI_Bcast(atomicindx , natmax*nspecies*ntrain, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(atomcount  , ntrain*nspecies       , MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(trrange    , ntrain                , MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(natoms     , ntrain                , MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(totalsizes , ntrain                , MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(kernsizes  , ntrain                , MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(atomspe    , ntrain*natmax         , MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(specarray  , M                     , MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(almax      , nspecies              , MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(ancut      , nspecies*(llmax+1)    , MPI_INT, 0, MPI_COMM_WORLD);

/////////////////

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
  if (nproc == 0) {
    t = MPI_Wtime ();
  }

  double * AVEC = calloc(sizeof(double)*totsize, 1);
  double * BMAT = calloc(sizeof(double)*totsize*totsize, 1);

  double * Avec = calloc(sizeof(double)*totsize, 1);
  double * Bmat = calloc(sizeof(double)*totsize*totsize, 1);

  int div = ntrain/Nproc;
  int rem = ntrain%Nproc;
  int start[Nproc+1];
  start[0] = 0;
  for(int i=0; i<Nproc; i++){
    start[i+1] = start[i] + ((i<rem)?div+1:div);
  }
#if 0
  if (nproc == 0) {
    for(int i=0; i<Nproc; i++){
      printf("%d %d %d\n", start[i], start[i+1], start[i+1]-start[i]);
    }
  }
#endif

  do_work(start[nproc], start[nproc+1], atomicindx, atomcount, trrange, natoms, totalsizes, kernsizes , atomspe   , specarray , almax     , ancut     , Avec, Bmat);

  MPI_Reduce (Avec, AVEC, totsize,         MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce (Bmat, BMAT, totsize*totsize, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if(nproc == 0){
    t = MPI_Wtime () - t;
    fprintf(stderr, "t=%4.2lf\n", t);
    vec_print(totsize, AVEC, "\n", stdout);
    vec_print(totsize*totsize, BMAT, "\n", stdout);
  }

  free(AVEC);
  free(BMAT);
  free(Avec);
  free(Bmat);

  free( atomicindx);
  free( atomcount );
  free( trrange   );
  free( natoms    );
  free( totalsizes);
  free( kernsizes );
  free( atomspe   );
  free( specarray );
  free( almax     );
  free( ancut     );

  MPI_Finalize ();
  return 0;
}
