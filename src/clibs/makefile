USE_MPI = 1
USE_OMP = 0

EQUISTORE_PATH=$(shell python -c 'import metatensor; print(metatensor.__path__[0])')
EQUISTORE_INCL_PATH=$(EQUISTORE_PATH)/include/
EQUISTORE_LINK_PATH=$(EQUISTORE_PATH)/lib/

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

.IGNORE:
all: get_matrices.so warn

%.so: %.o get_matrices_work.o
	$(CC) $^ -L$(EQUISTORE_LINK_PATH) -l:libmetatensor.so -lstdc++ -shared -Wl,-rpath=$(EQUISTORE_LINK_PATH),-soname,$@ -o $@

get_matrices_work.o: get_matrices_work.cpp
	$(CC) $(DEF) $(OMPFLAG) $(W) -I$(EQUISTORE_INCL_PATH) $^ -c --std=c++11 -O2 -fPIC -o $@

%.o: %.c
	$(CC) $(DEF) $(OMPFLAG) $(W) $^ -c --std=gnu11 -O2 -fPIC -o $@

clean:
	rm -vf get_matrices.so get_matrices_work.o

warn:
	@echo
	@echo -e "\e[1;31mATTENTION!\e[0m" Build the library in the same environment you are going to use it in.
	@echo Current metatensor path:
	@echo $(EQUISTORE_PATH)
	@echo
