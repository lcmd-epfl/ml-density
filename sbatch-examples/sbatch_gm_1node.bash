#!/bin/bash -l

#SBATCH --job-name=bmat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --exclude=node55,node06,node12,node05

module purge

conda activate density
path=../GIT/SOURCE
#time $path/get_matrices_C.py b --config=config_elec_norm-zeroed.txt
time $path/get_matrices_C.py b --config=config_elec_ps-nonnormalized.txt
