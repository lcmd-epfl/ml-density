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
    double * Bmat);
