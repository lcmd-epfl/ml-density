# installation

```
git clone https://github.com/lcmd-epfl/ml-density/
cd ml-density/
```

## environment

* `cargo`
* use `environment.yml` or
```
conda create -n ml-density python=3.11
conda activate ml-density
conda install pip=25.1.1
conda install numpy=1.23
conda install scipy=1.10
conda install numba=0.60.0
pip install mpi4py==3.1.4 ase==3.22 wigners==0.3
pip install metatensor-core==0.1.14
pip install git+https://github.com/metatensor/featomic.git@4968429ece4f5fc7935047d0c2e813cac21018ab
```

## build

```
make -C SOURCE/clibs
```

## usage

in progress... see `sidechains_examples/` and `sbatch_examples`

## todo

* why are the new PS so big?
* improve readme
* add tests
* replace `print_progress` by `tqdm`
* ? add `rho-predictor` as a dependency 
