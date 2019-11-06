#include <Python.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>

#define HEADER_SIZE 128

static PyObject * convert(PyObject * self, PyObject * args)
{
  const char * fname, * foname;

  if (!PyArg_ParseTuple(args, "ss", &fname, &foname)) return NULL;

  struct stat st;
  stat(fname, &st);
  size_t size = st.st_size;

  FILE * f = fopen(fname, "r");
  if(!f){
    fprintf(stderr, "file ? %s\n", fname);
    return Py_BuildValue("i", 1);
  }

  double * alldata = malloc(size);
  fread(alldata, size, 1, f);
  fclose(f);

  FILE * fo = fopen(foname, "w");
  if(!fo){
    fprintf(stderr, "file ? %s\n", foname);
    return Py_BuildValue("i", 2);
  }

  double * data = alldata + HEADER_SIZE/sizeof(data[0]);
  for(int i=0; i<(size-HEADER_SIZE)/sizeof(data[0]); i++){
    fprintf(fo, "%.10e\n", data[i]);
  }
  fclose(fo);

  free(alldata);
  return Py_BuildValue("i", 0);
}

static PyMethodDef npy2dat_methods[] =
{
  {"convert", convert, METH_VARARGS, "convert simple .npy files to text"},
  {NULL, NULL, 0, NULL}
};

/* module initialization --- Python version 2 */
  PyMODINIT_FUNC
initnpy2dat(void)
{
  (void) Py_InitModule("npy2dat", npy2dat_methods);
}

