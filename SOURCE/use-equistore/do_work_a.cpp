//#include "mylib.h"
#include <stdlib.h>
#include <stdio.h>
#include "eq/equistore.hpp"
#include <iostream>
using namespace equistore;

#define MAX_PATH_LENGTH 512

extern "C"{
#include "mylib_for_cpp.h"
}


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
    const unsigned int conf,
    const unsigned int * const atomcount, // number of elements
    const ao_t         * const aoref,     // number of elements
    const char * const path_proj,
    const char * const path_kern,
    double * Avec){

  char file_proj[MAX_PATH_LENGTH], file_kern[MAX_PATH_LENGTH];
  sprintf(file_proj, "%s%d.npz", path_proj, conf);
  sprintf(file_kern, "%s%d.dat.npz", path_kern, conf);

  auto ktensor = TensorMap::load(file_kern);
  auto ptensor = TensorMap::load(file_proj);

#pragma omp parallel shared(Avec)
#pragma omp for schedule(dynamic)
  for(int i1=0; i1<totsize; i1++){
    int iiref1 = aoref[i1].iiref;
    int im1    = aoref[i1].im;
    int n1     = aoref[i1].n;
    int l1     = aoref[i1].l;
    int a1     = aoref[i1].a;
    int q1     = aoref[i1].q;
    int msize1 = 2*l1+1;
    if(!atomcount[a1]){
      continue;
    }
    auto kvalues = get_matching_block_values(&ktensor, l1, q1);
    auto pvalues = get_matching_block_values(&ptensor, l1, q1);

    double dA = 0.0;
    for(int icel1=0; icel1<atomcount[a1]; icel1++){
      for(int imm1=0; imm1<msize1; imm1++){
        dA += pvalues(icel1, imm1, n1) * kvalues(icel1, imm1, im1, iiref1);
      }
    }
    Avec[i1] += dA;
  }
  return;
}
