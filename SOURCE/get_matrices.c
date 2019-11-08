#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define printalive printf("alive @ %s:%d\n", __FILE__, __LINE__)
#define GOTOHELL { \
  fprintf(stderr, "%s:%d %s() -- ", \
      __FILE__, __LINE__, __FUNCTION__); \
  abort(); }

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

static inline int sparseind(int i, int j, int k, int l, int natoms) {
  return
    i * natoms*nnmax*(llmax+1) +
    j * nnmax*(llmax+1) +
    k * (llmax+1) +
    l;
}

static inline int ksparseind(int i, int j, int k, int l, int m, int natoms) {
  return
    i * natoms * (2*llmax+1) * (llmax+1) * M +
    j * (2*llmax+1) * (llmax+1) * M +
    k * (llmax+1) * M +
    l * M +
    m;
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



int main(int argc, char ** argv){

  int * atomicindx = ivec_read(natmax*nspecies*ntrain  ,  "atomicindx_training.dat"    );
  int * atomcount  = ivec_read(ntrain*nspecies         ,  "atom_counting_training.dat" );
  int * trrange    = ivec_read(ntrain                  ,  "train_configs.dat"          );
  int * natoms     = ivec_read(ntrain                  ,  "natoms_train.dat"           );
  int * totalsizes = ivec_read(ntrain                  ,  "total_sizes.dat"            );
  int * kernsizes  = ivec_read(ntrain                  ,  "kernel_sizes.dat"           );
  int * atomspe    = ivec_read(ntrain*natmax           ,  "atomic_species.dat"         );
  int * specarray  = ivec_read(M                       ,  "fps_species.dat"            );
  int * almax      = ivec_read(nspecies                ,  "almax.dat"                  );
  int * ancut      = ivec_read(nspecies*(llmax+1)      ,  "anmax.dat"                  );


  double * Avec = calloc(sizeof(double)*totsize, 1);
  double * Bmat = calloc(sizeof(double)*totsize*totsize, 1);

  for(int itrain=0; itrain<ntrain; itrain++){
    printf("%d\n", itrain);

    //   ! read projections, overlaps and kernels for each training molecule
    int conf = trrange[itrain];
    char file_proj[512], file_over[512], file_kern[512];
    sprintf(file_proj, "%s%d.dat", path_proj, conf);
    sprintf(file_over, "%s%d.dat", path_over, conf);
    sprintf(file_kern, "%s%d.dat", path_kern, conf);

    double * projections = vec_read(totalsizes[itrain], file_proj);
    double * overlaps    = vec_read(totalsizes[itrain]*totalsizes[itrain], file_over);
    double * kernels     = vec_read(kernsizes[itrain], file_kern);
    int * sparseindexes = calloc(sizeof(int) * (2*llmax+1)*natoms[itrain]*nnmax*(llmax+1), 1);
    int * kernsparseindexes = calloc(sizeof(int) * (2*llmax+1)*natoms[itrain]*(2*llmax+1)*(llmax+1)*M, 1);


    int it1 = 0;
    for(int iat=0; iat<natoms[itrain]; iat++){
      int a1 = atomspe[itrain*natmax+iat];
      int al1 = almax[a1];
      for(int l1=0; l1<al1; l1++){
        int msize1 = 2*l1+1;
        int anc1   = ancut[ a1 * (llmax+1) + l1 ];
        for(int n1=0; n1<anc1; n1++){
          for(int im1=0; im1<msize1; im1++){
            sparseindexes[ sparseind(im1,iat,n1,l1, natoms[itrain])] = it1++;
          }
        }
      }
    }

    //vec_print(totalsizes[itrain]*totalsizes[itrain], overlaps, "\n", stdout);
    //vec_print(totalsizes[itrain], projections, "\n", stdout);

    int ik1 = 0;
    for(int iref1=0; iref1<M; iref1++){
      int a1 = specarray[iref1];
      int al1 = almax[a1];

      for(int l1=0; l1<al1; l1++){
        int msize1 = 2*l1+1;
        for(int im1=0; im1<msize1; im1++){
          for(int iat=0; iat<atomcount[ itrain*nspecies+a1 ]; iat++){
            for(int imm1=0; imm1<msize1; imm1++){
              kernsparseindexes[ ksparseind(imm1,iat,im1,l1,iref1, natoms[itrain])] = ik1++;
            }
          }
        }
      }
    }

    //   ! Loop over 1st dimension
    int i1 = 0;
    for(int iref1=0; iref1<M; iref1++){
      printf("%d / %d \n", iref1, M);
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
              for(int imm1=0; imm1<msize1; imm1++){
                int sp1 = sparseindexes    [ sparseind (imm1,iat,n1,l1, natoms[itrain])];
                int sk1 = kernsparseindexes[ ksparseind(imm1,icspe1,im1,l1,iref1, natoms[itrain])];
                Avec[i1] += projections[sp1] * kernels[sk1];
              }
            }

            //                   ! Loop over 2nd dimension
            int i2 = 0;
            for(int iref2=0; iref2<=iref1; iref2++){
              int a2 = specarray[iref2];
              int al2 = almax[a2];
              for(int l2=0; l2<al2; l2++){
                int msize2 = 2*l2+1;
                int anc2   = ancut[ a2 * (llmax+1) + l2 ];
                for(int n2=0; n2<anc2; n2++){
                  for(int im2=0; im2<msize2; im2++){
                    // ! Collect contributions for 1st dimension
                    double contrB = 0.0;
                    for(int icspe1=0; icspe1<atomcount[itrain*nspecies+a1]; icspe1++){
                      int iat = atomicindx[icspe1*nspecies*ntrain + a1*ntrain + itrain];
                      for(int imm1=0; imm1<msize1; imm1++){
                        int sp1 = sparseindexes    [ sparseind (imm1,iat,n1,l1, natoms[itrain])];
                        int sk1 = kernsparseindexes[ ksparseind(imm1,icspe1,im1,l1,iref1, natoms[itrain])];
                        //  ! Collect contributions for 2nd dimension
                        double Btemp = 0.0;
                        for(int icspe2=0; icspe2<atomcount[itrain*nspecies+a2]; icspe2++){
                          int jat = atomicindx[icspe2*nspecies*ntrain + a2*ntrain + itrain];
                          for(int imm2=0; imm2<msize2; imm2++){
                            int sp2 = sparseindexes    [ sparseind (imm2,jat,n2,l2, natoms[itrain])];
                            int sk2 = kernsparseindexes[ ksparseind(imm2,icspe2,im2,l2,iref2, natoms[itrain])];
                            Btemp += overlaps[sp2*totalsizes[itrain]+sp1] * kernels[sk2];
                          }
                        }
                        contrB += Btemp * kernels[sk1];
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

  //vec_print(totsize, Avec, "\n", stdout);
  vec_print(totsize*totsize, Bmat, "\n", stdout);

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

  return 0;
}
