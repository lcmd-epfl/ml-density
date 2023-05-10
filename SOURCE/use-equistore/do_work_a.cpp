//#include "mylib.h"
#include <stdlib.h>
#include <stdio.h>
#include "eq/equistore.hpp"
#include <iostream>
using namespace equistore;


typedef struct {
  int im;
  int n;
  int l;
  int a;
  int iref;
} ao_t;


extern "C" void do_work_a(
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
    const unsigned int *atomicindx,//[nelem][natmax],
    const unsigned int *atomcount  ,//[nelem],
    const unsigned int *atom_elem  ,//[natmax],
    const unsigned int *ref_elem   ,//[M],
    const unsigned int *alnum      ,//[nelem],
    const unsigned int **annum     ,//[nelem][llmax+1],
    const ao_t         *aoref      ,//[totsize],
    const char * const path_proj,
    const char * const path_kern,
    double * Avec);


equistore::NDArray<double> get_matching_block_values(TensorMap * tensor, int l, int q){
    auto matching_blocks_id = tensor->blocks_matching(Labels({"spherical_harmonics_l", "species_center"}, {{l, q}}));
    if(matching_blocks_id.size() != 1){
      abort();
    }
    auto block = tensor->block_by_id(matching_blocks_id[0]);
    return block.values();
}


void do_work_a(
    const unsigned int totsize,
    const unsigned int nelem,
    const unsigned int * elements,
    const unsigned int llmax, //
    const unsigned int nnmax, //
    const unsigned int M,     //
    const unsigned int natmax, //
    const unsigned int nat, //
    const unsigned int conf,
    const unsigned int nao, //
    const unsigned int kernsize, //
    const unsigned int *atomicindx,//
    const unsigned int *atomcount  ,//[nelem],
    const unsigned int *atom_elem  ,//
    const unsigned int *ref_elem   ,//
    const unsigned int *alnum      ,//
    const unsigned int **annum     ,//
    const ao_t         *aoref      ,//[totsize],
    const char * const path_proj,
    const char * const path_kern,
    double * Avec){


  char file_proj[512], file_kern[512];
  sprintf(file_proj, "%s%d.npz", path_proj, conf);
  sprintf(file_kern, "%s%d.dat.npz", path_kern, conf);

  auto ktensor = TensorMap::load(file_kern);
  auto ptensor = TensorMap::load(file_proj);

  int * iiref1 = (int *)calloc(nelem, sizeof(int));
  for(int i=0; i<nelem; i++){
    iiref1[i]=-1;
  }
  int pred_iref1 = -2;

//#pragma omp parallel shared(Avec)
//#pragma omp for schedule(dynamic)
  for(int i1=0; i1<totsize; i1++){
    int iref1 = aoref[i1].iref;
    int im1   = aoref[i1].im;
    int n1    = aoref[i1].n;
    int l1    = aoref[i1].l;
    int a1    = aoref[i1].a;
    int msize1 = 2*l1+1;
    if(pred_iref1 != iref1){
      pred_iref1 = iref1;
      iiref1[a1]++;
    }

    if(!atomcount[a1]){
      continue;
    }
    int q1 = elements[a1];
    auto kvalues = get_matching_block_values(&ktensor, l1, q1);
    auto pvalues = get_matching_block_values(&ptensor, l1, q1);

    double dA = 0.0;
    for(int icel1=0; icel1<atomcount[a1]; icel1++){
      for(int imm1=0; imm1<msize1; imm1++){
        dA += pvalues(icel1, imm1, n1) * kvalues(icel1, imm1, im1, iiref1[a1]);
      }
    }
    Avec[i1] += dA;
  }
  free(iiref1);

  return;
}
