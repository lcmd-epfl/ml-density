# ML-density

SA-GPR model for electron density

Used in:
* https://doi.org/10.1063/5.0033326
* https://doi.org/10.1021/acs.jpclett.1c01425
* https://doi.org/10.1063/5.0055393

Based on:
* https://doi.org/10.1021/acscentsci.8b00551
* https://doi.org/10.1039/C9SC02696G

## Installation

```
git clone https://github.com/lcmd-epfl/ml-density/
cd ml-density/
git checkout use-rascaline-ps
```

### Environment

* install `cargo`
* install MPI and OpenMP
* use `environment.yml` or
```
conda env create -f environment.yml

```

or

```
conda create -n ml-density python=3.13
conda activate ml-density
conda install pip=25.2
conda install numpy=2.3
conda install scipy=1.16.2
conda install numba=0.62
conda install mpi4py=4.0
pip install ase==3.26 wigners==0.3
pip install metatensor-core==0.1.17
pip install featomic==0.6.3
pip install git+https://github.com/lcmd-epfl/Q-stack.git@master
```

### Build

If planning to use the C library:
```
make -C src/clibs
```

## Usage

in progress... see `sidechains_examples/` and `sbatch_examples`

## TODO

* improve readme
* add tests
* replace `print_progress` by `tqdm`
* ? add `rho-predictor` as a dependency 
* fix OMP
