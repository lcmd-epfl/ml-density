# Background

## Density

The density is represented as a sum of atomic contributions (density fitting):

$$
\rho(\vec r) = \sum_i c_i \chi_i,
$$

where $[\chi_i]$ is a density-fitting basis such as `cc-pvqz-jkfit` and $[c_i]$ is a coefficients vector.
The index $i$ is a shorthand for $(\ell,n,m)$, where $\ell$ is the angular momentum, 
$m$ is the magnetic quantum number (for a real spherical harmonic), 
and $n$ is the radial channel index (since in a basis there can be several functions
with the same $(\ell, m)$ but different exponents).

## Prediction

The coefficients are obtained from kernel $K$ and weights $\vec x$:

$$
c_i^{\mathrm{ML}}(\vec x) = \sum_j K_{ij} x_j.
$$

## Kernel 

We use the $λ$-SOAP kernel. It is computed as a dot product of power spectra (representations).
The kernel can be interpreted as a function comparing orbitals of the same $\ell$ and $n$ but possibly different $m$
sitting on atoms (environments) with the same nuclear charge $q$.
Thus the full matrix $K$ has a block structure and consists of small $(2\ell+1) \times (2\ell+1)$ matrices.

## Reference environments

There are too many orbitals in the training set to compute and invert the regression matrix.
Thus the sum index $j$ in the predictions goes not over all the training set orbitals,
but over the orbitals sitting on a selected subset of atoms (reference environments).
Normally, there are $\sim 1000$ environments selected by farthest point sampling (FPS).


## Regression

The loss function is a sum of 
squared coefficient errors norms 
computed with a metrix matrix $O$
for each molecule:

$$
\sum_{\mathrm{train}} (\vec c-\vec c^{\mathrm{ML}})^T O (\vec c-\vec c^{\mathrm{ML}}).
$$

The matrix $O$ guides the optimization target (e.g. residue self-overlap or self-repulsion).
The regression equation becomes

$$
B \vec x = \vec A,
$$

where for historical reasons we call the coefficient matrix "B matrix" and the constant vector "A vector".
The problem dimensionality equals to the number of orbitals in the reference environments,
and $A$ and $B$ are computed as a sum of sparse matrix multiplications:

$$
B = \sum_{\mathrm{mol} \in \mathrm{train}} K_{\mathrm{refs}/\mathrm{mol}} O_{\mathrm{mol}} K_{\mathrm{mol}/\mathrm{refs}}
$$

$$
\vec A = \sum_{\mathrm{mol} \in \mathrm{train}} K_{\mathrm{refs}/\mathrm{mol}} O_{\mathrm{mol}}  \vec c_{\mathrm{mol}} 
$$

The computation of the $B$ matrix is the most computationally expensive step.

The weight are obtained by solving the linear system. For regularization, 
a unit matrix as well as a sparse kernel matrix $K_{refs/refs}$ are added to the $B$ matrix,
each multiplied by its own parameter (a non-negative number $\ll 1$).
