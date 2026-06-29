path=../src

# Predict on the testing set
$path/prediction.py
$path/compute_error.py > error.txt

# Predict on extra/1.xyz
$path/ex_power_spectra.py
$path/ex_kernels.py
$path/extrapolation.py
