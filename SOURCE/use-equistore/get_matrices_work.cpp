#include <equistore.hpp>
using namespace equistore;

extern "C"{
#include "get_matrices.h"
}

static inline size_t mpos(size_t i, size_t j){
  /* A[i+j*(j+1)/2], i <= j, 0 <= j < N */
  return (i)+(((j)*((j)+1))>>1);
}

#define ANNUM(A, L)                                     (annum[(A)*(llmax+1)+L])

#define KBLOCK(L,A)                                     ((L)*nelem+(A))
#define KVALUE(L,A,ICEL,IMM,IM,IIREF)                   ((((ICEL)*(2*(L)+1) + IMM)*(2*(L)+1) + IM)*nref[A] + IIREF)

#define OBLOCK(L1,L2,A1,A2)                             ((((L1)*(llmax+1) + L2)*nelem + A1)*nelem + A2)
#define OVALUE(L1,L2,A1,A2,ICEL1,ICEL2,IMM1,IMM2,N1,N2) (((((ICEL1 * atomcount[A2]) + ICEL2) * (2*L1+1) + IMM1) * (2*L2+1) + IMM2) * ANNUM(A1,L1) + N1) * ANNUM(A2,L2) + N2

#define KVAL(L,A,ICEL,IMM,IM,IIREF)                     kvalues[KBLOCK(L, A)][KVALUE(L, A, ICEL, IMM, IM, IIREF)]
#define OVAL(L1,L2,A1,A2,ICEL1,ICEL2,IMM1,IMM2,N1,N2)   ovalues[OBLOCK(L1,L2,A1,A2)][OVALUE(L1,L2,A1,A2,ICEL1,ICEL2,IMM1,IMM2,N1,N2)]

equistore::NDArray<double> get_matching_block_values(TensorMap * tensor, int l, int q){
  auto matching_blocks_id = tensor->blocks_matching(Labels({"spherical_harmonics_l", "species_center"}, {{l, q}}));
  if(matching_blocks_id.size() != 1){
    printf("%d\n", matching_blocks_id.size());
    abort();
  }
  auto block = tensor->block_by_id(matching_blocks_id[0]);
  return block.values();
}

equistore::NDArray<double> get_matching_block_values2(TensorMap * tensor, int l1, int l2, int q1, int q2){
  auto matching_blocks_id = tensor->blocks_matching(Labels({"spherical_harmonics_l1", "spherical_harmonics_l2", "species_center1", "species_center2"}, {{l1, l2, q1, q2}}));
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


void do_work_b(
    const unsigned int totsize,
    const unsigned int nelem,
    const unsigned int llmax,
    const unsigned int conf,
    const unsigned int * const atomcount ,//[nelem],
    const unsigned int * const nref      ,//[nelem]
    const unsigned int * const elements  ,//[nelem]
    const unsigned int * const alnum     ,//[nelem],
    const unsigned int * const annum     ,//[nelem][llmax+1],
    const ao_t         * const aoref     ,//[totsize],
    const char * const path_over,
    const char * const path_kern,
    double * Bmat){

  char file_over[MAX_PATH_LENGTH], file_kern[MAX_PATH_LENGTH];
  sprintf(file_over, "%s%d.npz", path_over, conf);
  sprintf(file_kern, "%s%d.dat.npz", path_kern, conf);
  auto ktensor = TensorMap::load(file_kern);
  auto otensor = TensorMap::load(file_over);

  double ** kvalues = (double **)calloc(nelem*(llmax+1), sizeof(double *));
  double ** ovalues = (double **)calloc(nelem*nelem*(llmax+1)*(llmax+1), sizeof(double *));
  for(int a1=0; a1<nelem; a1++){
    if(atomcount[a1]){
      for(int l1=0; l1<alnum[a1]; l1++){
        kvalues[KBLOCK(l1,a1)] = get_matching_block_values(&ktensor, l1, elements[a1]).data();
        for(int a2=0; a2<nelem; a2++){
          if(atomcount[a2]){
            for(int l2=0; l2<alnum[a2]; l2++){
              ovalues[OBLOCK(l1, l2, a1, a2)] = get_matching_block_values2(&otensor, l1, l2, elements[a1], elements[a2]).data();
            }
          }
        }
      }
    }
  }

#pragma omp parallel shared(Bmat)
#pragma omp for schedule(dynamic)
  for(int i1=0; i1<totsize; i1++){
    int iiref1 = aoref[i1].iiref;
    int im1    = aoref[i1].im;
    int n1     = aoref[i1].n;
    int l1     = aoref[i1].l;
    int a1     = aoref[i1].a;
    int msize1 = 2*l1+1;
    if(!atomcount[a1]){
      continue;
    }

    for(int i2=i1; i2<totsize; i2++){
      int iiref2 = aoref[i2].iiref;
      int im2    = aoref[i2].im;
      int n2     = aoref[i2].n;
      int l2     = aoref[i2].l;
      int a2     = aoref[i2].a;
      int msize2 = 2*l2+1;
      if(!atomcount[a2]){
        continue;
      }

      double dB = 0.0;
      for(int icel1=0; icel1<atomcount[a1]; icel1++){
        for(int imm1=0; imm1<msize1; imm1++){
          double Btemp = 0.0;
          for(int icel2=0; icel2<atomcount[a2]; icel2++){
            for(int imm2=0; imm2<msize2; imm2++){
              double o = OVAL(l1, l2, a1, a2, icel1, icel2, imm1, imm2, n1, n2);
              double k2 = KVAL(l2, a2, icel2, imm2, im2, iiref2);
              Btemp += o * k2;
            }
          }
          double k1 = KVAL(l1, a1, icel1, imm1, im1, iiref1);
          dB += Btemp * k1;
        }
      }
      Bmat[mpos(i1,i2)] += dB;
    }
  }
  free(kvalues);
  free(ovalues);
  return;
}
