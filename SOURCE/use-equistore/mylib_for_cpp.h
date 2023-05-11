typedef struct {
  int im;
  int n;
  int l;
  int a;
  int q;
  int iref;
  int iiref;
} ao_t;

void do_work_a(
    const unsigned int totsize,
    const unsigned int conf,
    const unsigned int * const atomcount,
    const ao_t         * const aoref,
    const char * const path_proj,
    const char * const path_kern,
    double * Avec);


int * kernsparseindices_fill(
    const int nat,
    const int llmax ,
    const int M,
    const unsigned int * const atomcount,  // nelem
    const unsigned int * const ref_elem,   // M
    const unsigned int * const alnum       // nelem
    );



int * sparseindices_fill(
    const int nat,
    const int llmax,
    const int nnmax,
    const unsigned int * const alnum,           //  nelem
    const unsigned int * const annum,  //  nelem*(llmax+1)
    const unsigned int * const atom_elem        //  natmax
    );

double * vec_readtxt(int n, const char * fname);

double * npy_read(int n, const char * fname);


void do_work_b(
    const unsigned int totsize,
    const unsigned int nelem,
    const unsigned int llmax,
    const unsigned int nnmax,
    const unsigned int M,
    const unsigned int natmax,
    const unsigned int nat,
    const unsigned int conf,
    const unsigned int nao,
    const unsigned int kernsize,
    const unsigned int * const atomicindx,//[nelem][natmax],
    const unsigned int * const atomcount ,//[nelem],
    const unsigned int * const atom_elem ,//[natmax],
    const unsigned int * const ref_elem  ,//[M],
    const unsigned int * const alnum     ,//[nelem],
    const unsigned int * const annum     ,//[nelem][llmax+1],
    const ao_t         * const aoref     ,//[totsize],
    const char * const path_over,
    const char * const path_kern,
    double * Bmat);
