.PHONY: all clean f2py

F2PY='/software/anaconda/5.2/python-2.7/bin/f2py'
FCOMPILER='gfortran'
#FCOMPILER='intelem' # gfortran
F90FLAGS='-fopenmp'
#F90FLAGS='-qopenmp' # -fopenmp
F2PYOPT='-O3'
#F2PYOPT='-O3 -unroll-aggressive -qopt-prefetch -qopt-reportr5'
LIBS='-lgomp'
#LIBS='-liomp5 -lpthread' # omp libraries with ifort

f2py: rmatrix.so regression.so prediction.so 

rmatrix.so: rmatrix.f90
	$(F2PY) -c --opt=$(F2PYOPT) rmatrix.f90    -m rmatrix --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS) 

regression.so: regression.f90
	$(F2PY) -c --opt=$(F2PYOPT) regression.f90 -m regression --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS)  $(LIBS)

prediction.so: prediction.f90
	$(F2PY) -c --opt=$(F2PYOPT) prediction.f90 -m prediction --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS)

all: f2py

clean:
	rm -fv *.o *.so
