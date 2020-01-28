#include <stdlib.h>
#include <stdio.h>

#define printalive printf("alive @ %s:%d\n", __FILE__, __LINE__)
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

ao_t * ao_fill(
    const int totsize ,
    const int llmax   ,
    const int M       ,
    const unsigned int const ref_elem[M],
    const unsigned int const alnum[],         // nelem
    const unsigned int const annum[][llmax+1] // nelem*(llmax+1)
    );

double * vec_readtxt(int n, const char * fname);

