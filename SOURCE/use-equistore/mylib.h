#include <stdlib.h>
#include <stdio.h>

#define printalive printf("alive @ %s:%d\n", __FILE__, __LINE__)

#define GOTOHELL { \
  fprintf(stderr, "%s:%d %s() -- ", \
      __FILE__, __LINE__, __FUNCTION__); \
  abort(); }

#define KSPARSEIND(IREF,L,ICSPE) ( ((IREF) * (llmax+1) + (L) ) * nat + (ICSPE) )
#define SPARSEIND(L,N,IAT) ( ((L) * nnmax + (N)) * nat + (IAT) )

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

ao_t * ao_fill(
    const int totsize ,
    const int llmax   ,
    const int M       ,
    const unsigned int const ref_elem[M],
    const unsigned int const alnum[],         // nelem
    const unsigned int const annum[][llmax+1] // nelem*(llmax+1)
    );

double * vec_readtxt(int n, const char * fname);


int * kernsparseindices_fill(
    const int nat,
    const int llmax ,
    const int M,
    const unsigned int const atomcount[],  // nelem
    const unsigned int const ref_elem[M],
    const unsigned int const alnum[]       // nelem
    );



void do_work_a(
    const unsigned int totsize,
    const unsigned int nelem,
    const unsigned int * elements,
    const unsigned int llmax,
    const unsigned int nnmax,
    const unsigned int M,
    const unsigned int natmax,
    const unsigned int nat,
    const unsigned int conf,
    const unsigned int nao,
    const unsigned int kernsize,
    const unsigned int const atomicindx[nelem][natmax],
    const unsigned int const atomcount [nelem],
    const unsigned int const atom_elem [natmax],
    const unsigned int const ref_elem  [M],
    const unsigned int const alnum     [nelem],
    const unsigned int const annum     [nelem][llmax+1],
    const ao_t         const aoref     [totsize],
    const char * const path_proj,
    const char * const path_kern,
    double * Avec);


int * sparseindices_fill(
    const int nat,
    const int llmax,
    const int nnmax,
    const unsigned int const alnum[],           //  nelem
    const unsigned int const annum[][llmax+1],  //  nelem*(llmax+1)
    const unsigned int const atom_elem[]        //  natmax
    );
