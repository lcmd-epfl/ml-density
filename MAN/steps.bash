module purge
module load anaconda/2019.3/python-3.7
module load intelmpi/17.0.4

path=SOURCE

# 2) Define sparse set {M} from scalar SOAP-vector

echo === ENVIRONMENTS
time python3 $path/environments.py
echo

# 2.5) Extract PS for {M}

echo === POWER SPECTRA FOR REFERENCE ENVIRONMENTS
time python3 $path/power_spectra_ref.py
echo

# 3) Build K_NM

echo === KERNELS
time python3 $path/kernels.py
echo

# 4) Build K_MM

echo === K_MM
time python3 $path/rmatrix.py
echo

# 5) Reorder QM data for l=1

echo === REORDER
time python3 $path/reorder_ao.py
echo

# 6) Baseline coefficients and compute projections

echo === BASELINE AND PROJECT
time python3 $path/project.py
echo

# 6.5) Make a training set

echo === TRAINING SET SELECTION
python3 $path/training_selection.py
echo

# 7) Compute A and B

echo === MAKE K*S*K AND K*W
time mpirun -np 9 python3 $path/get_matrices.py
time mpirun -np 9 python3 $path/get_matrices.py b
echo

# 8) REGRESSION!

echo === INVERT MATRIX
time python $path/regression.py
echo

# 9) predictions

echo === PREDICTIONS ON THE TEST SET
time python $path/prediction.py
echo

# 10) compute error

time python $path/compute_error.py

