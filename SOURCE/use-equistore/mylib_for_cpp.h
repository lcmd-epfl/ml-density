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
