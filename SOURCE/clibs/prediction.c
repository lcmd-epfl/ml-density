#include "mylib.h"
#include <math.h>

int prediction(
    const int nelem ,
    const int llmax ,
    const int nnmax ,
    const int M     ,
    const int ntest ,
    const int natmax,
    const unsigned int const atomicindx[ntest][nelem][natmax],
    const unsigned int const atomcount [ntest][nelem],
    const unsigned int const testrange [ntest],
    const unsigned int const natoms    [ntest],
    const unsigned int const kernsizes [ntest],
    const unsigned int const ref_elem  [M],
    const unsigned int const alnum     [nelem],
    const unsigned int const annum     [nelem][llmax+1],
    double w[M][llmax+1][nnmax][2*llmax+1],
    double coeffs[ntest][natmax][llmax+1][nnmax][2*llmax+1],
    const char * const path_kern){

#pragma omp parallel for schedule(dynamic)
  for(int itest=0; itest<ntest; itest++){

    const int nat = natoms[itest];
    const int conf = testrange[itest];
    char file_kern[512];
    sprintf(file_kern, "%s%d.dat", path_kern, conf);
    double * kernels  = vec_readtxt(kernsizes[itest], file_kern);

    int * kernsparseindices = calloc(sizeof(int)*nat*(2*llmax+1), 1);

    int ik = 0;
    for(int iref=0; iref<M; iref++){
      int a = ref_elem[iref];
      int al = alnum[a];
      for(int l=0; l<al; l++){
        int msize = 2*l+1;
        for(int im=0; im<msize; im++){
          for(int icel=0; icel<atomcount[itest][a]; icel++){
            kernsparseindices[icel*(2*llmax+1)+im] = ik;
            ik += msize;
          }
        }
        int anc = annum[a][l];
        for(int icel=0; icel<atomcount[itest][a]; icel++){
          int iat = atomicindx[itest][a][icel];
          for(int n=0; n<anc; n++){
            for(int imm=0; imm<msize; imm++){
              double d = 0.0;
              for(int im=0; im<msize; im++){
                int sk = kernsparseindices[icel*(2*llmax+1)+im];
                d += kernels[sk+imm] * w[iref][l][n][im];
              }
              coeffs[itest][iat][l][n][imm] += d;
            }
          }
        }
      }
    }
    free(kernsparseindices);
    free(kernels);
  }
  return 0;
}

