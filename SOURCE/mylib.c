#include "mylib.h"

ao_t * ao_fill(
    const int totsize ,
    const int llmax   ,
    const int M       ,
    const unsigned int const ref_elem[M],
    const unsigned int const alnum[],         // nelem
    const unsigned int const annum[][llmax+1] // nelem*(llmax+1)
    ){

  ao_t * aoref = malloc(sizeof(ao_t)*totsize);

  int i = 0;
  for(int iref=0; iref<M; iref++){
    int a = ref_elem[iref];
    int al = alnum[a];
    for(int l=0; l<al; l++){
      int msize = 2*l+1;
      int anc   = annum[a][l];
      for(int n=0; n<anc; n++){
        for(int im=0; im<msize; im++){
          aoref[i].im = im;
          aoref[i].n  = n;
          aoref[i].l  = l;
          aoref[i].a  = a;
          aoref[i].iref = iref;
          i++;
        }
      }
    }
  }
  return aoref;
}

double * vec_readtxt(int n, const char * fname){

  double * v = malloc(sizeof(double)*n);
  FILE   * f = fopen(fname, "r");
  if(!f){
    GOTOHELL;
  }
  for(int i=0; i<n; i++){
    if(fscanf(f, "%lf", v+i)!=1){
      GOTOHELL;
    }
  }
  fclose(f);
  return v;
}
