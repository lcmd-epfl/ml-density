USE_MPI = 1
USE_OMP = 1
EQUISTORE_PATH=./equistore

CC=gcc
OMPFLAG=
DEF=
ifeq ($(USE_MPI),1)
	DEF := $(DEF) -DUSE_MPI
	CC=mpicc
endif
ifeq ($(USE_OMP),1)
	OMPFLAG = -fopenmp
endif

W= -Warray-bounds -Wmaybe-uninitialized -Wmissing-braces -Wparentheses -Wsequence-point -Wtype-limits -Wundef \
   -Wuninitialized -Wunused -Wmisleading-indentation -Wempty-body -Wunused-but-set-variable -Wunused-parameter -Winline

all: get_matrices.so

%.so: %.o get_matrices_work.o
	$(CC) $^ -L$(EQUISTORE_PATH) -l:libequistore.a -lstdc++ -shared -Wl,-soname,$@ -o $@

get_matrices_work.o: get_matrices_work.cpp
	$(CC) $(DEF) $(OMPFLAG) $(W) -I$(EQUISTORE_PATH) $^ -c --std=c++11 -O2 -fPIC -o $@

%.o: %.c
	$(CC) $(DEF) $(OMPFLAG) $(W) $^ -c --std=gnu11 -O2 -fPIC -o $@

clean:
	rm -vf get_matrices.so get_matrices_work.o

