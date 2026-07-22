#!/bin/bash -l
#SBATCH --job-name=qm7
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=350G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/qm7-%j.out
#
# Full SA-GPR / lambda-SOAP pipeline for the QM7 set, full_gpr=true.
# Runs training -> prediction -> extrapolation for the src/*.py pipeline.
#
# One config file per basis selects everything (basis name, metric input dir, coeffs
# filename tag, and the INNER_<basis>/ output prefix); the shared inputs (compounds/,
# computed/, dataset.csv, extra/) are read-only and common to all three:
#     config_dz.txt   -> cc-pvdz-jkfit      -> metric/      -> INNER/
#     config_tz.txt   -> cc-pvtz-jkfit      -> metric_tz/   -> INNER_tz/
#     config_aqz.txt  -> aug-cc-pvqz-jkfit  -> metric_aqz/  -> INNER_aqz/
#
# Submit from THIS directory (QM7_test) so the relative paths in the config resolve.
#
# Prerequisites (done once per basis):
#   (1) conda env `ml-density`  (conda env create -f ../environment.yml)
#   (2) metric matrices unpacked to square:  python3 convert_metric.py --basis <basis>
#       -> metric/ (dz), metric_tz/ (tz), metric_aqz/ (aqz)

# --- environment (this cluster hides conda behind a loader; module/conda not on default PATH) ---
source /etc/profile.d/software.sh    # defines the cluster condald loader
condald                              # put `conda` on PATH
conda activate ml-density

# The `#!/bin/bash -l` login shell sources ~/.profile, which does `cd $HOME`.
# Return to the directory sbatch was invoked from (must be QM7_test) so ../src and the
# relative data paths (compounds/, metric*/, INNER*/, logs/) in the config resolve.
cd "${SLURM_SUBMIT_DIR:-$PWD}"

src=../src
cfg=$1
[ -f "$cfg" ] || { echo "config '$cfg' not found (expected config_{dz,tz,aqz}.txt; pass basis as arg 1)" >&2; exit 1; }
echo "[run] config=${cfg}  job=${SLURM_JOB_NAME:-?}  jobid=${SLURM_JOB_ID:-?}"
echo "Config file:"
echo ${cfg}
# The conda env's mpi4py is built against conda-forge MPICH (its own Hydra mpirun), which is
# NOT srun/PMI-aware -- launch the MPI stages with mpirun, not srun, or ranks won't see each other.
MPI="mpirun -np ${SLURM_NTASKS:-1}"

set -euo pipefail

# --- timing helpers: run_step wraps each stage, logs its duration to stdout (-> SLURM log),
# and at the end prints the total wall time and the slowest step. ---
STEP_NAMES=()
STEP_TIMES=()

fmt_dur() {
    local s=$1
    printf '%02d:%02d:%02d' $((s/3600)) $((s%3600/60)) $((s%60))
}

run_step() {
    local name="$1" cmd="$2" status=0 start=$SECONDS
    echo "[TIMING] >>> ${name} starting at $(date '+%Y-%m-%d %H:%M:%S')"
    eval "$cmd" || status=$?
    local dur=$((SECONDS-start))
    STEP_NAMES+=("$name")
    STEP_TIMES+=("$dur")
    printf '[TIMING] <<< %-28s %s (%ds)\n' "$name" "$(fmt_dur "$dur")" "$dur"
    if [ "$status" -ne 0 ]; then
        echo "[TIMING] step '${name}' failed with exit code ${status}"
        return $status
    fi
}

print_timing_summary() {
    local max_name="" max_dur=-1
    for i in "${!STEP_NAMES[@]}"; do
        if [ "${STEP_TIMES[$i]}" -gt "$max_dur" ]; then
            max_dur=${STEP_TIMES[$i]}
            max_name=${STEP_NAMES[$i]}
        fi
    done
    echo "[TIMING] ==================================================="
    echo "[TIMING] TOTAL: $(fmt_dur "$SECONDS") (${SECONDS}s)"
    echo "[TIMING] Slowest step: ${max_name} -- $(fmt_dur "$max_dur") (${max_dur}s,$(awk "BEGIN {printf \"%.1f\", ($max_dur / $SECONDS) * 100}")%)"
}

# --- training pipeline (full GPR) ---
run_step "preprocess"                    "$src/preprocess.py --config=$cfg"                         # baseline coeffs, process metric, projections
run_step "training_selection"            "$src/training_selection.py --config=$cfg"                 # random train/test split (seed, train_size)
run_step "power_spectra"                 "$MPI $src/power_spectra.py --config=$cfg"                 # lambda-SOAP descriptors (MPI)
run_step "select_reference_environments" "$src/select_reference_environments.py --config=$cfg"      # FPS sparse set of M references
run_step "power_spectra_reference"       "$src/power_spectra_reference.py --config=$cfg"            # reference power spectra
run_step "kernel_mm"                     "$src/kernel_mm.py --config=$cfg"                          # K_MM
run_step "kernel_nm"                     "$MPI $src/kernel_nm.py --config=$cfg"                     # K_NM (MPI)
run_step "get_matrices"                  "$MPI $src/get_matrices.py -b --config=$cfg"               # full GPR: -b builds A and B (MPI)
run_step "regression"                    "$src/regression.py --config=$cfg"                         # solve -> weights (+ PITC Cholesky)
run_step "prediction"                    "$src/prediction.py --config=$cfg"                         # predict test-set coefficients
run_step "variance"                      "$src/variance.py --config=$cfg"                           # PITC predictive variance
run_step "compute_error"                 "$src/compute_error.py --config=$cfg > error_gpr_${SLURM_JOB_NAME:-?}.txt"

# --- extrapolation / out-of-sample (extra/qm7_extra.xyz) ---
run_step "power_spectra_extra" "$MPI $src/power_spectra.py --extra --config=$cfg"
run_step "kernel_nm_extra"     "$MPI $src/kernel_nm.py --extra --config=$cfg"
run_step "prediction_extra"    "$src/prediction.py --extra --config=$cfg"
run_step "variance_extra"      "$src/variance.py --extra --config=$cfg"

print_timing_summary
