.PHONY: all clean f2py

F2PY='/software/anaconda/5.2/python-2.7/bin/f2py'
FCOMPILER='gfortran'
#FCOMPILER='intelem' # gfortran
#F90FLAGS='-fopenmp'
#F90FLAGS='-qopenmp' # -fopenmp
F2PYOPT='-O3'
#F2PYOPT='-O3 -unroll-aggressive -qopt-prefetch -qopt-reportr5'
LIBS='-lgomp'
#LIBS='-liomp5 -lpthread' # omp libraries with ifort

MODULES=rmatrix.so prediction.so

all: $(MODULES)

%.so: %.f90
	$(F2PY) -c --opt=$(F2PYOPT) $^ -m $(@:%.so=%) --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS)

clean:
	rm -fv $(MODULES)
