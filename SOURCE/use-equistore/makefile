USE_MPI = 1
USE_OMP = 1

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

%.so: %.o mylib.o do_work_a.o
	$(CC) $^ -L./eq -l:libequistore.a -lstdc++ -shared -Wl,-soname,$@ -o $@

do_work_a.o: do_work_a.cpp
	$(CC) $(DEF) $(OMPFLAG) $(W) $^ -c --std=c++11 -O2 -fPIC -o $@

%.o: %.c
	$(CC) $(DEF) $(OMPFLAG) $(W) $^ -c --std=gnu11 -O2 -fPIC -o $@

clean:
	rm -vf get_matrices.so
	rm -vf mylib.o do_work_a.o

