#include "mylib.h"

ao_t * ao_fill(
    const int nelem   ,
    const int totsize ,
    const int llmax   ,
    const int M       ,
    const unsigned int const elements[nelem],
    const unsigned int const ref_elem[M],
    const unsigned int const alnum[],         // nelem
    const unsigned int const annum[][llmax+1] // nelem*(llmax+1)
    ){

  ao_t * aoref = malloc(sizeof(ao_t)*totsize);

  int * iiref = (int *)calloc(nelem, sizeof(int));
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
          aoref[i].q  = elements[a];
          aoref[i].iref = iref;
          aoref[i].iiref = iiref[a];
          i++;
        }
      }
    }
    iiref[a]++;
  }
  free(iiref);
  return aoref;
}

double * vec_readtxt(int n, const char * fname){

  double * v = malloc(sizeof(double)*n);
  FILE   * f = fopen(fname, "r");
  if(!f){
    fprintf(stderr, "cannot open file %s", fname);
    GOTOHELL;
  }
  for(int i=0; i<n; i++){
    if(fscanf(f, "%lf", v+i)!=1){
      fprintf(stderr, "cannot read file %s line %d", fname, i+1);
      GOTOHELL;
    }
  }
  fclose(f);
  return v;
}


int * kernsparseindices_fill(
    const int nat,
    const int llmax ,
    const int M,
    const unsigned int const atomcount[],  // nelem
    const unsigned int const ref_elem[M],
    const unsigned int const alnum[]       // nelem
    ){

  int * kernsparseindices = calloc(M*(llmax+1)*nat, sizeof(int));
  int i = 0;
  for(int iref=0; iref<M; iref++){
    int a = ref_elem[iref];
    int al = alnum[a];
    for(int l=0; l<al; l++){
      int msize = 2*l+1;
      for(int iat=0; iat<atomcount[a]; iat++){
        kernsparseindices[KSPARSEIND(iref,l,iat)] = i;
        i += msize*msize;
      }
    }
  }
  return kernsparseindices;
}

int * sparseindices_fill(
    const int nat,
    const int llmax,
    const int nnmax,
    const unsigned int const alnum[],           //  nelem
    const unsigned int const annum[][llmax+1],  //  nelem*(llmax+1)
    const unsigned int const atom_elem[]        //  natmax
    ){

  int * sparseindices = calloc((llmax+1) * nnmax * nat, sizeof(int));
  int i = 0;
  for(int iat=0; iat<nat; iat++){
    int a = atom_elem[iat];
    int al = alnum[a];
    for(int l=0; l<al; l++){
      int msize = 2*l+1;
      int anc   = annum[a][l];
      for(int n=0; n<anc; n++){
        sparseindices[ SPARSEIND(l,n,iat)] = i;
        i += msize;
      }
    }
  }
  return sparseindices;
}
