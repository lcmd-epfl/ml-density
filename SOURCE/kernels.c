#include <stdlib.h>
#include <stdio.h>
#include <math.h>

static double vecdot(size_t n, double * u, double * v){
  double s = 0.0;
// https://computing.llnl.gov/tutorials/openMP/#SHARED
#pragma omp parallel for      \
  default(shared)             \
  schedule(static,1024)      \
  reduction(+:s)
  for(size_t i=0; i<n; i++){
    s += u[i]*v[i];
  }
  return s;
}

int kernels(

int M
,int natoms
,int llmax
,int l
,int iref
,int iatspe
,double zeta

,unsigned int kernel_sparse_indexes[M][natoms][llmax+1][2*llmax+1][2*llmax+1]

,int len
,double * powert
,double * powerr

,double * k_NM

){

  if(l==0){
    int ik = kernel_sparse_indexes[iref][iatspe][l][0][0];
    k_NM[ik] = vecdot(len,powert,powerr);
    k_NM[ik] = pow(k_NM[ik], zeta);
  }
  else{

    int ik0 = kernel_sparse_indexes[iref][iatspe][0][0][0];
    double mult = pow(k_NM[ik0], (zeta-1.0/zeta));
    for(int im1=0; im1<2*l+1; im1++){
      for(int im2=0; im2<2*l+1; im2++){
        int ik = kernel_sparse_indexes[iref][iatspe][l][im2][im1];
        k_NM[ik] = mult * vecdot(len, powert+len*im1, powerr+len*im2);
      }
    }


  }

  return 0;
}
