#include <stdlib.h>
#include <stdio.h>

#define GOTOHELL { \
  fprintf(stderr, "%s:%d %s() -- ", \
      __FILE__, __LINE__, __FUNCTION__); \
  abort(); }

static inline size_t mpos(size_t i, size_t j){
  /* A[i+j*(j+1)/2], i <= j, 0 <= j < N */
  return (i)+(((j)*((j)+1))>>1);
}

static inline size_t symsize(size_t M){
  return (M*(M+1))>>1;
}

typedef struct {
  int im;
  int n;
  int l;
  int a;
  int iref;
} ao_t;

static void vec_read(size_t n, double * v, const char * fname){
  FILE * f = fopen(fname, "r");
  if(!f){
    GOTOHELL;
  }
  if(fread(v, sizeof(double), n, f) != n){
    GOTOHELL;
  }
  fclose(f);
  return;
}

int make_matrix(
    const int n,
    const int llmax,
    const int M,
    const int * const specarray ,  // M
    const int * const almax     ,  // nspecies
    const int * const ancut     ,  // nspecies*(llmax+1)
    const double kMM[llmax+1][M*(2*llmax+1)][M*(2*llmax+1)],
    double * mat, double reg, double jit, char * Bmatpath){

  ao_t * ao = malloc(sizeof(ao_t)*n);

  int i = 0; // stolen from get_matrices()
  for(int iref=0; iref<M; iref++){
    int a = specarray[iref];
    int al = almax[a];
    for(int l=0; l<al; l++){
      int msize = 2*l+1;
      int anc   = ancut[a*(llmax+1)+l];
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

  double * Bmat = malloc(sizeof(double)*symsize(n));
  vec_read(symsize(n), Bmat, Bmatpath);

  for(size_t j=0; j<n; j++){
    int  n1  = ao[j].n;
    int  l1  = ao[j].l;
    int  a1  = ao[j].a;
    int iref1 = ao[j].iref;
    int  im1 = ao[j].im;
    int msize = 2*l1+1;
    int ik1 = iref1*msize+im1;
    for(size_t i=0; i<j; i++){
      double dmat = Bmat[mpos(i,j)];
      int  n2  = ao[i].n;
      int  l2  = ao[i].l;
      int  a2  = ao[i].a;
      if(a1==a2 && l1==l2 && n1==n2){
        int iref2 = ao[i].iref;
        int  im2 = ao[i].im;
        int ik2 = iref2*msize+im2;
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

