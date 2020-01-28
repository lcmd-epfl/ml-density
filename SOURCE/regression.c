#include "mylib.h"

static double * vec_read(size_t n, const char * fname){

  double * v = malloc(sizeof(double)*n);
  FILE   * f = fopen(fname, "r");
  if(!f){
    GOTOHELL;
  }
  if(fread(v, sizeof(double), n, f) != n){
    GOTOHELL;
  }
  fclose(f);
  return v;
}

int make_matrix(
    const int n,
    const int llmax,
    const int M,
    const unsigned int const ref_elem[M],      // M
    const unsigned int const alnum[],          // nelem
    const unsigned int const annum[][llmax+1], // nelem*(llmax+1)
    const double kMM[llmax+1][M*(2*llmax+1)][M*(2*llmax+1)],
    double * mat, double reg, double jit, char * Bmatpath){

  ao_t * ao = ao_fill(n, llmax, M, ref_elem, alnum, annum);

  double * Bmat = vec_read(symsize(n), Bmatpath);

  for(size_t j=0; j<n; j++){
    int n1    = ao[j].n;
    int l1    = ao[j].l;
    int a1    = ao[j].a;
    int iref1 = ao[j].iref;
    int im1   = ao[j].im;
    int msize = 2*l1+1;
    int ik1   = iref1*msize+im1;
    for(size_t i=0; i<j; i++){
      double dmat = Bmat[mpos(i,j)];
      int n2  = ao[i].n;
      int l2  = ao[i].l;
      int a2  = ao[i].a;
      if(a1==a2 && l1==l2 && n1==n2){
        int iref2 = ao[i].iref;
        int im2   = ao[i].im;
        int ik2   = iref2*msize+im2;
        dmat += reg*kMM[l1][ik1][ik2];
      }
      mat[i*n+j] = mat[j*n+i] = dmat;
    }
    mat[j*n+j] = Bmat[mpos(j,j)] + reg*kMM[l1][ik1][ik1] + jit;
  }

  free(Bmat);
  free(ao);
  return 0;
}

