#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define GOTOHELL { \
  fprintf(stderr, "%s:%d %s() -- ", \
      __FILE__, __LINE__, __FUNCTION__); \
  abort(); }

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

int prediction(
    const int nspecies ,
    const int llmax    ,
    const int nnmax    ,
    const int M        ,
    const int ntest    ,
    const int natmax   ,
    const int * const atomicindx,  //  ntest*nspecies*natmax
    const int * const atomcount ,  //  ntest*nspecies
    const int * const testrange ,  //  ntest
    const int * const natoms    ,  //  ntest
    const int * const kernsizes ,  //  ntest
    const int * const specarray ,  //  M
    const int * const almax     ,  //  nspecies
    const int * const ancut     ,  //  nspecies*(llmax+1)
    double w[M][llmax+1][nnmax][2*llmax+1],
    double coeffs[ntest][natmax][llmax+1][nnmax][2*llmax+1],
    const char * const path_kern){

#pragma omp parallel for schedule(dynamic)
  for(int itest=0; itest<ntest; itest++){

    const int nat = natoms[itest];
    const int conf = testrange[itest];
    char file_kern[512];
    sprintf(file_kern, "%s%d.dat", path_kern, conf);
    double * kernels  = vec_read(kernsizes[itest], file_kern);

    int * kernsparseindexes = calloc(sizeof(int)*nat*(2*llmax+1), 1);

    int ik = 0;
    for(int iref=0; iref<M; iref++){
      int a = specarray[iref];
      int al = almax[a];
      for(int l=0; l<al; l++){
        int msize = 2*l+1;
        for(int im=0; im<msize; im++){
          for(int icspe=0; icspe<atomcount[itest*nspecies+a]; icspe++){
            kernsparseindexes[icspe*(2*llmax+1)+im] = ik;
            ik += msize;
          }
        }
        int anc   = ancut[ a * (llmax+1) + l ];
        for(int icspe=0; icspe<atomcount[itest*nspecies+a]; icspe++){
          int iat = atomicindx[itest*nspecies*natmax+a*natmax+icspe];
          for(int n=0; n<anc; n++){
            for(int imm=0; imm<msize; imm++){
              double d = 0.0;
              for(int im=0; im<msize; im++){
                int sk = kernsparseindexes[icspe*(2*llmax+1)+im];
                d += kernels[sk+imm] * w[iref][l][n][im];
              }
              coeffs[itest][iat][l][n][imm] += d;
            }
          }
        }
      }
    }
    free(kernsparseindexes);
    free(kernels);
  }
  return 0;
}

