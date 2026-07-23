#!/bin/bash -l
#
# Usage: bash slurm.bash [basis_name] (dz (default), tz or aqz)

basis=${1:-dz}
ntasks=${2:-48} # 48, 36, 12 for dz, tz, aqz respectively is recommended.
sbatch --job-name $basis --output logs/$basis-%j.out --ntasks $ntasks run.bash config_$basis.txt
