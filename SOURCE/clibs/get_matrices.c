#include <math.h>
#ifdef USE_MPI
#include <mpi.h>
#include <string.h>
#endif
#include "get_matrices.h"

int Nproc;
int nproc;

static inline size_t symsize(size_t M){
  return (M*(M+1))>>1;
}

static ao_t * ao_fill(
    const int nelem   ,
    const int totsize ,
    const int M       ,
    const unsigned int * const elements,  // nelem
    const unsigned int * const ref_elem,  // M
    const unsigned int * const alnum,     // nelem
    const unsigned int * const annum      // (llmax1)*nelem
    ){

  ao_t * aoref = malloc(sizeof(ao_t)*totsize);

  int * iiref = (int *)calloc(nelem, sizeof(int));
  int i = 0;
  for(int iref=0; iref<M; iref++){
    int a = ref_elem[iref];
    int al = alnum[a];
    for(int l=0; l<al; l++){
      int msize = 2*l+1;
      int anc   = annum[l*nelem+a];
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


static void print_mem(const int totsize, const int ntrain, FILE * f){

  const double b2mib = 1.0/(1<<20);
  const double b2gib = 1.0/(1<<30);
  size_t size = (symsize(totsize))*sizeof(double);

  fprintf(f, "\
      Problem dimensionality = %d\n\
      Number of training molecules = %d\n\
      output: %16zu bytes (%10.2lf MiB, %6.2lf GiB)\n\n",
      totsize, ntrain, size, size*b2mib, size*b2gib);
  fflush(f);
  return;
}

#ifdef USE_MPI
static void print_nodes(FILE * f){

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  char buf[256];
  snprintf(buf, sizeof(buf), " proc %4d : %s\n", nproc, processor_name);

  if(nproc){
    MPI_Send(buf, strlen(buf)+1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
  }
  else{
    fputs(buf, f);
    MPI_Status status;
    for(int i=1; i<Nproc; i++) {
      MPI_Recv(buf, sizeof(buf), MPI_CHAR, i, 0, MPI_COMM_WORLD, &status);
      fputs(buf, f);
    }
    fputs("\n", f);
    fflush(f);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  return;
}

static void send_jobs(const int bra, const int ket){
  for(int imol=bra; imol<ket+(Nproc-1); imol++){
    MPI_Status status;
    int p[2];
    MPI_Recv(p, 2, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
    p[1] = (imol<ket) ? imol : -1;
    MPI_Send(p, 2, MPI_INT, p[0], 0, MPI_COMM_WORLD);
    printf("%4d: %4d\n", p[0], p[1]);
  }
  return;
}

static int receive_job(int imol){
  int p[] = {nproc, imol};
  MPI_Status status;
  MPI_Send(p, 2, MPI_INT, 0, 0, MPI_COMM_WORLD);
  MPI_Recv(p, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
  return p[1];
}

#endif

static void print_batches(FILE * f, int nfrac, const unsigned int * const ntrains, const char ** const paths){
  if(!nproc){
    for(int i=0; i<nfrac; i++){
      fprintf(f, "batch %2d [%d--%d):\t %s\n", i, (i==0?0:ntrains[i-1]), ntrains[i], paths[i]);
    }
    fprintf(f, "\n");
    fflush(f);
  }
#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
}

static void vec_print(size_t n, double * v, const char * mode, const char * fname){

  FILE * f = fopen(fname, mode);
  if(!f){
    fprintf(stderr, "cannot open file %s", fname);
    GOTOHELL;
  }
  for(int i=0; i<n; i++){
    fprintf(f, "% 18.15e\n", v[i]);
  }
  fprintf(f, "\n");
  fclose(f);
  return;
}

static void vec_write(size_t n, double * v, const char * mode, const char * fname){

  FILE * f = fopen(fname, mode);
  if(!f){
    fprintf(stderr, "cannot open file %s", fname);
    GOTOHELL;
  }
  if(fwrite(v, sizeof(double), n, f) != n){
    fprintf(stderr, "cannot write to file %s line %d", fname);
    GOTOHELL;
  }
  fclose(f);
  return;
}


int get_a(
    const unsigned int totsize,
    const unsigned int nelem,
    const unsigned int M,
    const unsigned int ntrain,
    const unsigned int nfrac,
    const unsigned int const ntrains  [nfrac],
    const unsigned int const atomcount[ntrain][nelem],
    const unsigned int const trrange  [ntrain],
    const unsigned int const ref_elem [M],
    const unsigned int * const alnum,
    const unsigned int * const annum,
    const unsigned int const elements[nelem],
    const char * const path_proj,
    const char * const path_kern,
    const char ** const paths_avec
    ){


#ifdef USE_MPI
  int argc = 1;
  char * argv0 = "get_matrices";
  char ** argv = &argv0;
  MPI_Init (&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD, &Nproc);
  MPI_Comm_rank (MPI_COMM_WORLD, &nproc);
  print_nodes(stdout);
#else
  nproc = 0;
  Nproc = 1;
#endif
  print_batches(stdout, nfrac, ntrains, paths_avec);

#ifdef USE_MPI
  double t = 0.0;
  if(nproc == 0){
    t = MPI_Wtime ();
  }
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  double * Avec = calloc(sizeof(double)*totsize, 1);
  ao_t * aoref = ao_fill(nelem, totsize, M, elements, ref_elem, alnum, annum);


  if(Nproc==1){
    for(int ifrac=0; ifrac<nfrac; ifrac++){
      for(int imol=(ifrac==0?0:ntrains[ifrac-1]); imol<ntrains[ifrac]; imol++){
        printf("%4d: %4d\n", nproc, imol);
        do_work_a(totsize, trrange[imol], atomcount[imol], aoref, path_proj, path_kern, Avec);
      }
      vec_print(totsize, Avec, "w", paths_avec[ifrac]);
    }
#ifdef USE_MPI
    t = MPI_Wtime () - t;
    fprintf(stderr, "t=%4.2lf\n", t);
#endif
  }

#ifdef USE_MPI
  else{

    double * AVEC = NULL;
    if(!nproc){
      AVEC = calloc(sizeof(double)*totsize, 1);
    }

    for(int ifrac=0; ifrac<nfrac; ifrac++){
      if(!nproc){
        send_jobs(ifrac==0?0:ntrains[ifrac-1], ntrains[ifrac]);
      }
      else{
        int imol = -1;
        while(1){
          imol = receive_job(imol);
          if(imol<0){
            break;
          }
          do_work_a(totsize, trrange[imol], atomcount[imol], aoref, path_proj, path_kern, Avec);
        }
        printf("%4d: finished work\n", nproc);
      }

      MPI_Barrier(MPI_COMM_WORLD);
      if(!nproc){
        double tt = MPI_Wtime();
        fprintf(stderr, "batch%d: t=%4.2lf\n", ifrac, tt-t);
        t = tt;
      }
      MPI_Reduce (Avec, nproc?NULL:AVEC, totsize, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      if(!nproc){
        vec_print(totsize, AVEC, "w", paths_avec[ifrac]);
      }
    }
    free(AVEC);
  }
#endif

  free(aoref);
  free(Avec);

#ifdef USE_MPI
  MPI_Finalize ();
#endif
  return 0;
}

int get_b(
    const unsigned int totsize,
    const unsigned int nelem  ,
    const unsigned int M      ,
    const unsigned int ntrain ,
    const unsigned int nfrac,
    const unsigned int const ntrains   [nfrac],
    const unsigned int const atomcount [ntrain][nelem],
    const unsigned int const trrange   [ntrain],
    const unsigned int const ref_elem  [M],
    const unsigned int const * alnum,
    const unsigned int const * annum,
    const unsigned int const elements  [nelem],
    const char * const path_over,
    const char * const path_kern,
    const char ** const paths_bmat
    ){

#ifdef USE_MPI
  int argc = 1;
  char * argv0 = "get_matrices";
  char ** argv = &argv0;
  MPI_Init (&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD, &Nproc);
  MPI_Comm_rank (MPI_COMM_WORLD, &nproc);
  print_nodes(stdout);
#else
  nproc = 0;
  Nproc = 1;
#endif
  print_batches(stdout, nfrac, ntrains, paths_bmat);

#ifdef USE_MPI
  size_t bufsize = (1<<30)/sizeof(double);  // number of doubles to take 1 GiB
  if(bufsize > symsize(totsize)){
    bufsize = symsize(totsize);
  }
  if(Nproc == 1){
    bufsize = 0;
  }

  double t = 0.0;
  if(nproc == 0){
    print_mem(totsize, ntrain, stdout);
    t = MPI_Wtime ();
  }
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  double * Bmat = calloc(sizeof(double)*symsize(totsize), 1);
  ao_t * aoref = ao_fill(nelem, totsize, M, elements, ref_elem, alnum, annum);
  int * nref = calloc(nelem, sizeof(int));
  for(int i=0; i<M; i++){
    nref[ref_elem[i]]++;
  }
  int llmax1 = 0;
  for(int a=0; a<nelem; a++){
    if(llmax1<alnum[a]){
      llmax1 = alnum[a];
    }
  }

  if(Nproc==1){
    for(int ifrac=0; ifrac<nfrac; ifrac++){
      for(int imol=(ifrac==0?0:ntrains[ifrac-1]); imol<ntrains[ifrac]; imol++){
        printf("%4d: %4d\n", nproc, imol);
        do_work_b(totsize, nelem, llmax1, trrange[imol], atomcount[imol], nref, elements, alnum, annum, aoref, path_over, path_kern, Bmat);
      }
      vec_write(symsize(totsize), Bmat, "w", paths_bmat[ifrac]);
    }
#ifdef USE_MPI
    t = MPI_Wtime () - t;
    fprintf(stderr, "t=%4.2lf\n", t);
#endif
  }

#ifdef USE_MPI
  else{

    double * BMAT = NULL;
    if(!nproc){
      BMAT = calloc(sizeof(double)*bufsize, 1);
    }

    for(int ifrac=0; ifrac<nfrac; ifrac++){
      if(!nproc){
        send_jobs(ifrac==0?0:ntrains[ifrac-1], ntrains[ifrac]);
      }
      else{
        int imol = -1;
        while(1){
          imol = receive_job(imol);
          if(imol<0){
            break;
          }
          do_work_b(totsize, nelem, llmax1, trrange[imol], atomcount[imol], nref, elements, alnum, annum, aoref, path_over, path_kern, Bmat);
        }
        printf("%4d: finished work\n", nproc);
      }

      MPI_Barrier(MPI_COMM_WORLD);
      if(!nproc){
        double tt = MPI_Wtime();
        fprintf(stderr, "batch%d: t=%4.2lf\n", ifrac, tt-t);
        t = tt;
      }

      size_t div = symsize(totsize)/bufsize;
      size_t rem = symsize(totsize)%bufsize;
      for(size_t i=0; i<=div; i++){
        size_t size = i<div?bufsize:rem;
        if(!size){
          break;
        }
        MPI_Reduce (Bmat+i*bufsize, nproc?NULL:BMAT, size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if(!nproc){
          vec_write(size, BMAT, i?"a":"w", paths_bmat[ifrac]);
          printf("chunk #%d/%d written\n", i+1, rem?(div+1):div);
        }
      }
    }
    free(BMAT);
  }
#endif

  free(nref);
  free(aoref);
  free(Bmat);

#ifdef USE_MPI
  MPI_Finalize ();
#endif
  return 0;
}
