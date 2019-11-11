#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>

#define HEADER_SIZE 128

int convert(const char * fname, const char * foname){

  struct stat st;
  stat(fname, &st);
  size_t size = st.st_size;

  FILE * f = fopen(fname, "r");
  if(!f){
    fprintf(stderr, "file ? %s\n", fname);
    return 1;
  }

  double * alldata = malloc(size);
  fread(alldata, size, 1, f);
  fclose(f);

  FILE * fo = fopen(foname, "w");
  if(!fo){
    fprintf(stderr, "file ? %s\n", foname);
    return 2;
  }

  double * data = alldata + HEADER_SIZE/sizeof(data[0]);
  for(int i=0; i<(size-HEADER_SIZE)/sizeof(data[0]); i++){
    fprintf(fo, "%.10e\n", data[i]);
  }
  fclose(fo);

  free(alldata);
  return 0;
}

