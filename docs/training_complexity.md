# Training: matrix operations and computational complexity

This note walks through **every linear-algebra operation performed during training**
of the SA-GPR electron-density model, gives each one a complexity, and identifies
the bottleneck.

Training is split across two scripts:

1. **`src/get_matrices.py -b`** — assembles the **Gram matrix** and the **target vector**
   (one streaming pass over the training molecules).
2. **`src/regression.py`** — solves one symmetric positive-definite linear system for the
   **regression weights**.

The code has three training modes, selected by the config option `regression_model`
([`config_paths.py`](../src/libs/config_paths.py)):

- `regression_model = gpr_DTC` → **DTC** (Deterministic Training Conditional) — **the model of
  main.pdf**, and the one this note leads with. A sparse GP whose training system is *identical* to
  SA-GPR's, plus a posterior.
- `regression_model = sagpr` → **SA-GPR**, the deterministic symmetry-adapted fit (historically a
  KRR). The default; it produces a prediction and no uncertainty at all, and is summarized for
  contrast at the end.
- `regression_model = gpr_PITC` → **PITC** (Partially Independent Training Conditional) — a
  stricter sparse GP that additionally carries the Nyström residual $\mathbf{D}_i$. **Not part of
  main.pdf**; kept in the code as a reference point, and documented after DTC.

All three are *sparse / inducing-point* fits over `M` reference environments; none ever forms the
dense `N × N` kernel over all training atoms. So the distinction is **not** sparse-versus-full — the
retired `full_gpr` name suggested otherwise — but which model is being fitted. Each is derived on
its own terms — DTC from main.pdf Eqs. 21–22, PITC from its own $\mathbf{\Lambda}_i$ — and the
family table below maps how they relate.

> **A word on SoR.** The Subset of Regressors approximation is a sparse *GP* whose linear system
> reduces exactly to the one SA-GPR already solves (main.pdf §II D 1). That coincidence is a result,
> not an identity: SoR also carries a predictive variance,
> $\sigma_p^2\lambda\mathbf{V}^\top\mathbf{V}$, which is degenerate — it collapses far from the
> reference set. SA-GPR produces no variance at all, so it is not SoR, and SoR proper is **not
> implemented here**. DTC is the repair of that degeneracy, at identical cost.

---

## Notation

| Symbol | Meaning | Code name / location |
|--------|---------|----------------------|
| $N$ | number of training molecules | loop bound over `ntrains` ([`get_matrices.py:26`](../src/get_matrices.py)) |
| $M$ | number of reference (inducing) environments | `o.M` ([`config_paths.py:32`](../src/libs/config_paths.py)) |
| $n_i$ | number of density-fitting AO coefficients for molecule $i$ | `nao_i` ([`pitc_lib.py:82`](../src/libs/pitc_lib.py)) |
| $\bar n$ | average $n_i$; $\sum_i n_i$ = total training AO coefficients | — |
| $P$ | **problem dimensionality** = total AO coefficients over the $M$ references = size of the weight vector and of the Gram matrix | `nao_ref = basis.nao_for_mol(ref_elements)` ([`regression.py:24`](../src/regression.py), [`gram_matrix.py:245`](../src/libs/gram_matrix.py)) |
| $\lambda$ | regularization parameter, $\sigma_\rho^2/\sigma_p^2$ (main.pdf Eq. 17) | `o.reg`, config `regularisation` ([`config_paths.py`](../src/libs/config_paths.py)) |
| $\sigma_p^2$ | prior variance; the overall factor on the predictive variance | `o.sigma_p2`, config `prior_scale` ([`config_paths.py`](../src/libs/config_paths.py)) |

Key size relations (used in the bottleneck analysis):

$$
P \;\approx\; M \cdot n_{\text{atom}}, \qquad
\sum_i n_i \;=\; N\,\bar n \;=\; N \cdot A_{\text{mol}} \cdot n_{\text{atom}},
$$

where $n_{\text{atom}}$ is the average number of auxiliary AOs per atom
(`nao_atom`, [`functions.py:138`](../src/libs/functions.py)) and $A_{\text{mol}}$ is the
average number of atoms per molecule. Because the training set has far more atoms than
the $M$ references, in practice $\sum_i n_i \gg P$.

The Gram matrix is stored **packed as a flattened lower triangle** of length
$\tfrac{P(P+1)}{2}$ (`symsize`, [`gram_matrix.py:46`](../src/libs/gram_matrix.py)).

---

## Inputs to training (built by earlier pipeline stages)

These are read from disk during training, not recomputed, but they define the operands:

- $\mathbf{K}_{MM}$ — reference↔reference kernel, $(P \times P)$, built by
  `kernel_mm.py` ([`kernels_lib.py:121`](../src/libs/kernels_lib.py)).
- $\mathbf{K}_{I_i M}$ — molecule↔reference kernel, $(n_i \times P)$, built by
  `kernel_nm.py` ([`kernels_lib.py:9`](../src/libs/kernels_lib.py)).
- $\mathbf{K}_{I_i I_i}$ — molecule self-kernel, $(n_i \times n_i)$.
- $\mathbf{S}_i$ — per-molecule Coulomb **metric** (overlap) matrix, $(n_i \times n_i)$,
  often ill-conditioned (κ ≈ 10⁷), hence diagonal jitter ([`pitc_lib.py`](../src/libs/pitc_lib.py)).
  The jitter is **relative** to each matrix's mean diagonal (`jitter_scale`): $\mathbf{\Sigma}_M$'s
  diagonal reaches ~10⁹ while $\mathbf{K}_{MM}$'s is $O(1)$, so one absolute value cannot serve both.
- $\mathbf{w}_i$ — metric-projected density-fitting coefficients (the fit target),
  $(n_i,)$.

Kernels use a **ζ = 2** symmetry-adapted (λ-SOAP) nonlinearity: each $\lambda > 0$ block is
multiplied by the scalar $\lambda = 0$ kernel ([`kernels_lib.py:147`](../src/libs/kernels_lib.py)).

---

## DTC training (`regression_model = gpr_DTC`) — the main.pdf model

DTC applies the Nyström replacement $\mathbf{K}_{AB}\to\mathbf{Q}_{AB}$ (Eq. 20) to the training
conditional only, leaving the test–test block $\mathbf{K}_{**}$ exact. Writing $\lambda$ for the
ratio $\sigma_\rho^2/\sigma_p^2$ of Eq. 17, the model is

$$\mathbf{A} = \mathbf{K}_{TR}^\top\mathbf{M}_T\mathbf{K}_{TR} + \lambda\,\mathbf{K}_{RR},\qquad
\boxed{\;\mathbf{A}\,\mathbf{c} \;=\; \mathbf{K}_{TR}^\top\mathbf{M}_T\,\Delta\tilde{\mathbf{c}}_T\;}\tag{Eq. 21}$$

with $\Delta\tilde{\mathbf{c}}_T = \tilde{\mathbf{c}}_T - \bar{\mathbf{c}}_T$ the baselined fitted
coefficients. Two features of this form drive the whole implementation.

**The metric appears uninverted.** The observation noise of Eq. 15 is
$\sigma_\rho^2\mathbf{M}_T^{-1}$, so the precision it contributes is $\mathbf{M}_T/\sigma_\rho^2$ —
the metric itself — and the $1/\lambda$ it carries is folded once into the $\lambda\mathbf{K}_{RR}$
term rather than divided out per molecule. So **no matrix is inverted or factorized during
assembly**: `gram_matrix.do_work_gram` contracts $\mathbf{K}^\top\mathbf{M}_i\mathbf{K}$ straight
from the stored metric, and `target_vector.do_work_target` consumes the projections
$\mathbf{w}_i = \mathbf{M}_i\mathbf{y}_i$ that `preprocess.py` already tabulated. Eq. 21 is the very
system SA-GPR solves, which `regression.fill_matrix` builds, so DTC's predicted density is SA-GPR's
to roundoff (measured: $2\times10^{-7}$ relative on the weights, sidechains/M=32 — the
Cholesky-vs-LDLᵀ difference at $\kappa(\mathbf{A})\approx10^{10}$).

**The mean does not contain $\sigma_p^2$.** Eq. 21 has no prior scale in it, which is what lets the
scale be determined *after* the solve, from quantities the solve already produced.

**What DTC adds over SA-GPR**, all of it after the assembly:

1. the Cholesky factor $\mathbf{L}$ of $\mathbf{A}$, persisted for `variance.py`;

2. the prior scale by type-II maximum likelihood (`dtc_lib.fit_prior_scale`). Eq. 25 maximizes the
   marginal likelihood in closed form and Woodbury reduces it to quantities already on disk:
   $\Delta\tilde{\mathbf{c}}_T^\top\mathbf{M}_T\Delta\tilde{\mathbf{c}}_T$ is `p.coef_norms[:, 1]`
   summed over the training set, and $\lVert\mathbf{y}\rVert^2 = \mathbf{t}^\top\mathbf{A}^{-1}\mathbf{t}$
   is the target vector dotted with the solved weights, so the fit costs one sum plus one inner
   product:

$$\sigma_p^2 = \frac{1}{\lambda N_T}\Big[\Delta\tilde{\mathbf{c}}_T^\top\mathbf{M}_T\Delta\tilde{\mathbf{c}}_T - \lVert\mathbf{y}\rVert^2\Big],
\qquad \mathbf{y} = \mathbf{L}^{-1}\mathbf{K}_{TR}^\top\mathbf{M}_T\Delta\tilde{\mathbf{c}}_T\tag{Eq. 26}$$

   `prior_scale = fit` (the default) uses this; a positive float in that option pins $\sigma_p^2$
   instead, and `regression.py` logs the estimate the pinned value overrides. Since Eq. 21 does not
   contain $\sigma_p^2$, pinning it never changes a prediction — only the width of the error bar.

3. the non-degenerate predictive covariance (`dtc_lib.predictive_covariance`), carrying $\lambda$
   explicitly because $\mathbf{L}$ factorizes $\mathbf{A}$:

$$\mathbf{\Sigma}_{c_*} = \sigma_p^2\big[\mathbf{K}_{**} - \mathbf{U}^\top\mathbf{U} + \lambda\,\mathbf{V}^\top\mathbf{V}\big],
\qquad \mathbf{U} = \mathbf{L}_{RR}^{-1}\mathbf{K}_{R*},\quad \mathbf{V} = \mathbf{L}^{-1}\mathbf{K}_{R*}\tag{Eq. 22}$$

The leading $\mathbf{K}_{**} - \mathbf{Q}_{**}$ residual is what survives from leaving the test–test
block exact, and is exactly what SA-GPR (and SoR) lack: it is why DTC's error bar grows back toward
the prior for a test environment poorly spanned by the reference set instead of collapsing there.
Its cost is SA-GPR's; the caveat is that DTC treats training and test environments inconsistently
and so is a valid predictive rule but not a Gaussian process.

### Choosing $\lambda$ (`regularisation = fit`)

Eq. 26 identifies only the **product** $\sigma_p^2\lambda$, which by Eq. 17 is $\sigma_\rho^2$, the
noise variance. Since DTC's only channel for what the $M$ inducing points cannot represent is that
noise, the bracket of Eq. 26 is the unexplained training residual and the closed form divides it by
$\lambda$. Measured on QM7/cc-pvdz-jkfit, $\sigma_p^2\lambda$ is constant to five significant
figures over $\lambda\in[10^{-8},1]$ ($1.3457\times10^{-3}$ at $M=128$, $9.8121\times10^{-4}$ at
$M=256$). So the estimate pins the *noise*, while Eq. 22 then uses $\sigma_p^2$ as the *prior* scale
against the $O(1)$ $\mathbf{K}_{**}-\mathbf{Q}_{**}$ term: choosing $\lambda$ by hand chooses the
reported error bar by hand. At $\lambda=10^{-6}$ the QM7 calibration
$\langle(\mathbf{c}-\mathbf{c}_0)^\top\mathbf{J}(\mathbf{c}-\mathbf{c}_0)\rangle/\langle\mathrm{Tr}(\mathbf{\Sigma}_{c_*}\mathbf{J})\rangle$
comes out at $1.6\times10^{-6}$, against PITC's $0.77$.

Setting `regularisation = fit` instead maximizes the marginal likelihood of Eq. 23 over $\lambda$
(`dtc_lib.fit_regularisation`, called from `regression.fit_reg`). With $\sigma_p^2$ profiled out at
its Eq. 25 optimum and $\mathbf{C}_1 = \mathbf{Q}_{TT} + \lambda\mathbf{M}_T^{-1}$,

$$-2\log p = N_T\log\sigma_p^2(\lambda) + \log|\mathbf{C}_1(\lambda)| + \text{const},\qquad
\log|\mathbf{C}_1| = (N_T-P)\log\lambda + \log|\mathbf{A}(\lambda)| - \log|\mathbf{K}_{RR}| - \sum_i\log|\mathbf{S}_i|$$

the last two terms being independent of $\lambda$ and therefore dropped. Each trial value costs one
Cholesky of $\mathbf{A}(\lambda)$, i.e. $O(P^3/3)$ — cheap because **DTC's Gram matrix and target
vector do not depend on $\lambda$ at all**, so nothing is re-assembled. A 9-point log grid over
$[10^{-8},1]$ is scanned and the bracketing interval refined by a bounded Brent search, about 15–20
factorizations in total; peak memory rises from two $P\times P$ matrices to three. The fitted value
goes to `p.fitted_reg` and is read back by `variance_lib.dtc_lambda`, since $\lambda$ appears
explicitly in Eq. 22. Output file names carry `regfit` in place of the numeric value.

> **Divergence from main.pdf Sec. II F**, which prescribes $\lambda$ "by a one-dimensional scan or
> cross-validation of the predictive mean". Eq. 23 is the paper's own marginal likelihood, and it is
> the only route with any grip on QM7: there the predictive mean is flat to three digits across all
> eight decades of $\lambda$ (baselined MAE $48.9\%$ at every point, $M=128$), so a CV scan has
> nothing to minimize.

This is a **gpr_PITC-incompatible** option, and `config_paths.py` rejects the combination: PITC's
$\mathbf{\Lambda}_i = \mathbf{D}_i + \eta\mathbf{S}_i^{-1}$ puts the parameter inside the assembly,
so scanning it would mean re-running `get_matrices.py` per trial value.

What it buys, measured: on sidechains (8 training molecules, $M=32$) the fit picks
$\lambda = 1.9\times10^{-3}$ and the baselined MAE falls from $131\%$ to $9.2\%$ (PITC gets
$27.8\%$) while the calibration goes from $3.4\times10^{-3}$ to $0.30$. On QM7 (2048 training
molecules) the Gram matrix dominates $\lambda\mathbf{K}_{RR}$ everywhere, so the fit is
accuracy-neutral — the baselined MAE is flat at $48.9\%$ ($M=128$) across all eight decades — and
the gain is purely in the error bar, $1.6\times10^{-6} \to 0.16$ at the fitted
$\lambda\approx10^{-1}$. The residual factor of $\sim6$ against a perfect $1.0$ is $\mathbf{D}_i$
proper: even at its best white-in-$\mathbf{S}$ noise level, DTC cannot represent a structured
residual. `tests/dtc_eta_scan.py` reproduces these curves for any completed gpr_DTC tree.

### Jitter

DTC keeps SA-GPR's *absolute* diagonal jitter (`o.jit` passed straight to `fill_matrix`), unlike the
relative jitter the GP models use elsewhere. That is deliberate: $\mathbf{A}$ is
near-singular by construction — measured on sidechains/M=32, mean diagonal $70$ but smallest
eigenvalue $6\times10^{-9}$ — so a jitter relative to the mean diagonal lands *above* the smallest
eigenvalue and regularizes the solution rather than merely stabilizing it (it moved the weights by
20%). `robust_cholesky` then starts at zero and escalates only if the factorization actually fails.

---

## The sparsification family

DTC and PITC differ in which blocks of the joint prior the Nyström replacement
$\mathbf{K}_{AB}\to\mathbf{Q}_{AB}$ (main.pdf Eq. 20) is applied to. DTC applies it to the whole
training conditional, which is what leaves $\mathbf{M}_T$ standing uninverted in
$\mathbf{A}$; PITC keeps the per-molecule residual
$\mathbf{D}_i = \mathbf{K}_{I_iI_i} - \mathbf{K}_{I_iM}\mathbf{K}_{MM}^{-1}\mathbf{K}_{MI_i}$,
which costs a metric inverse and a Cholesky per molecule:

| mode | system solved | per-molecule assembly cost | predictive variance |
|---|---|---|---|
| DTC | $\mathbf{A} = \mathbf{K}_{TR}^\top\mathbf{M}_T\mathbf{K}_{TR} + \lambda\mathbf{K}_{RR}$ (Eq. 21) | contract $\mathbf{K}^\top\mathbf{M}_i\mathbf{K}$; nothing factorized | Eq. 22, non-degenerate |
| SA-GPR | $\mathbf{A}$, the same matrix | the same contraction | none computed |
| PITC | $\mathbf{\Sigma}_M = \mathbf{K}_{RR} + \sum_i\mathbf{K}^\top\mathbf{\Lambda}_i^{-1}\mathbf{K}$, $\mathbf{\Lambda}_i = \mathbf{D}_i + \eta\mathbf{S}_i^{-1}$ | $\mathbf{D}_i$, $\mathbf{S}_i^{-1}$ and a $\mathbf{\Lambda}_i$ Cholesky | yes |

Setting $\mathbf{D}_i = 0$ turns PITC's system into $\eta$ times DTC's, which is what
`tests/test_dtc_scaling.py` checks. That is a relation between the two, not a definition of DTC:
`dtc_lib.py` implements Eqs. 21/22/26 directly and never goes through `pitc_lib.py`.

## PITC training (`regression_model = gpr_PITC`) — the stricter alternative

> PITC is **not part of main.pdf**. It is kept in the code as the stricter member of the same
> family: where DTC replaces the whole training conditional by its Nyström projection, PITC keeps
> the per-molecule residual $\mathbf{D}_i$, at the assembly cost documented below. It is also what
> `tests/test_dtc_scaling.py` checks DTC's assembly against.

### The math

For each training molecule $i$, PITC forms a per-molecule precision block $\mathbf{\Lambda}_i$
from the Nyström residual and the (scaled) metric inverse:

$$
\mathbf{D}_i \;=\; \mathbf{K}_{I_i I_i} - \mathbf{K}_{I_i M}\,\mathbf{K}_{MM}^{-1}\,\mathbf{K}_{M I_i},
\qquad
\mathbf{\Lambda}_i \;=\; \mathbf{D}_i + \eta\,\mathbf{S}_i^{-1}.
$$

These accumulate into a Gram matrix and target vector:

$$
\boxed{\;\mathbf{\Sigma}_M \;=\; \mathbf{K}_{MM} + \sum_{i=1}^{N} \mathbf{K}_{M I_i}\,\mathbf{\Lambda}_i^{-1}\,\mathbf{K}_{I_i M}\;}
\qquad
\boxed{\;\mathbf{t} \;=\; \sum_{i=1}^{N} \mathbf{K}_{M I_i}\,\mathbf{\Lambda}_i^{-1}\,\mathbf{y}_i\;}
$$

with $\mathbf{y}_i = \mathbf{c}_i - \mathbf{c}_{av}$ the baselined coefficients, read directly from
`p.clean_coefficients`. (Equivalently $\mathbf{S}_i^{-1}\mathbf{w}_i$, but recovering $\mathbf{y}_i$
that way costs a factorization and loses ~10 digits at $\kappa(\mathbf{S})\sim10^7$ — see
`do_work_target_pitc`'s docstring.)

and the weights are the solution of

$$
\boxed{\;\mathbf{\Sigma}_M\,\mathbf{c} \;=\; \mathbf{t} \;\;\Longrightarrow\;\; \mathbf{c} = \mathbf{\Sigma}_M^{-1}\,\mathbf{t}.\;}
$$

### Stage A — assemble $\mathbf{\Sigma}_M$ and $\mathbf{t}$  (`get_matrices.py -b`)

**Setup (once).** Densify $\mathbf{K}_{MM}$, add jitter, factorize:

| # | Operation | Kind | Shape | Complexity | Code |
|---|-----------|------|-------|-----------|------|
| A0 | $\mathbf{L}_{MM} = \operatorname{chol}(\mathbf{K}_{MM} + \text{jit}\cdot\bar{d}\,\mathbf{I})$ | Cholesky | $P \times P$ | $O(P^3)$ | [`pitc_lib.py`](../src/libs/pitc_lib.py) |

**Per training molecule $i$** (`molecule_lambda_chol_knm`, [`pitc_lib.py`](../src/libs/pitc_lib.py)):

| # | Operation | Kind | Shape | Complexity | Code |
|---|-----------|------|-------|-----------|------|
| A1 | $\mathbf{U}_i = \mathbf{L}_{MM}^{-1}\mathbf{K}_{M I_i}$ = `solve_triangular` | triangular solve, $n_i$ RHS | $P \times n_i$ | $O(P^2 n_i)$ | [`pitc_lib.py`](../src/libs/pitc_lib.py) |
| A2 | $\mathbf{D}_i = \mathbf{K}_{I_i I_i} - \mathbf{U}_i^{\top}\mathbf{U}_i$ (exactly symmetric) | matrix–matrix | $n_i \times n_i$ | $O(n_i^2 P)$ | [`pitc_lib.py`](../src/libs/pitc_lib.py) |
| A3 | $\mathbf{S}_i^{-1}$ = `cho_factor` + `cho_solve(I)` | Cholesky + solve → explicit inverse | $n_i \times n_i$ | $O(n_i^3)$ | [`pitc_lib.py`](../src/libs/pitc_lib.py) |
| A4 | $\mathbf{\Lambda}_i = \mathbf{D}_i + \eta\,\mathbf{S}_i^{-1}$ | add + symmetrize | $n_i \times n_i$ | $O(n_i^2)$ | [`pitc_lib.py`](../src/libs/pitc_lib.py) |
| A5 | $\mathbf{L}_i = \operatorname{chol}(\mathbf{\Lambda}_i)$ = `robust_cholesky` | Cholesky ($n_i^3/3$, vs $2n_i^3$ for the LU inverse it replaced) | $n_i \times n_i$ | $O(n_i^3)$ | [`pitc_lib.py`](../src/libs/pitc_lib.py) |
| A6 | $\mathbf{V}_i = \mathbf{L}_i^{-1}\mathbf{K}_{I_i M}$, then $\mathbf{G}_i = \mathbf{V}_i^{\top}\mathbf{V}_i$ | triangular solve + products in **column chunks** (no full $P\times P$ temporary) | temp $\le$ `chunk_bytes` | $O(n_i^2 P) + O(P^2 n_i)$ | [`gram_matrix.py`](../src/libs/gram_matrix.py) |
| A7 | scatter each chunk's columns into packed Gram | accumulate | $P(P{+}1)/2$ | $O(P^2)$ | [`gram_matrix.py`](../src/libs/gram_matrix.py) |
| A8 | $\mathbf{z}_i = \mathbf{L}_i^{-1}\mathbf{y}_i$, with $\mathbf{y}_i$ read from disk | triangular solve | $n_i$ | $O(n_i^2)$ | [`target_vector.py`](../src/libs/target_vector.py) |
| A9 | $\mathbf{t}_i = \mathbf{V}_i^{\top}\mathbf{z}_i$ | matrix–vector | $P$ | $O(P n_i)$ | [`target_vector.py`](../src/libs/target_vector.py) |

Because $P \gg n_i$ (the references span all $M$ atoms; $n_i$ is a single molecule), the
**per-molecule cost is dominated by A1 and A6, both $O(P^2 n_i)$**. Summed over the loop
([`gram_matrix.py:274`](../src/libs/gram_matrix.py)):

$$
\text{Stage A} \;=\; \underbrace{O(P^3)}_{\text{A0, once}} \;+\; \sum_{i=1}^{N} O(P^2 n_i)
\;=\; O(P^3) + O\!\Big(P^2 \textstyle\sum_i n_i\Big)
\;=\; O\!\big(P^2 \, N\,\bar n\big).
$$

MPI ranks each accumulate a partial Gram/target and are summed with `Reduce`
([`gram_matrix.py:303`](../src/libs/gram_matrix.py)).

### Stage B — the training solve (`regression.py`, `gpr_PITC` branch)

| # | Operation | Kind | Shape | Complexity | Code |
|---|-----------|------|-------|-----------|------|
| B0 | densify $\mathbf{K}_{MM}$ (`kmm_cholesky`) | densify | $P \times P$ | $O(P^2)$ | [`regression.py:33`](../src/regression.py) |
| B1 | `unravel_tril` — unpack packed Gram → dense | unpack | $P \times P$ | $O(P^2)$ | [`regression.py:38`](../src/regression.py) |
| B2 | $\mathbf{\Sigma}_M = \text{Gram} + \mathbf{K}_{MM}$ | add | $P \times P$ | $O(P^2)$ | [`regression.py:39`](../src/regression.py) |
| B3 | $\mathbf{L} = \operatorname{chol}(\mathbf{\Sigma}_M)$ (`robust_cholesky`, ×10 escalation from `jit`·mean-diagonal) | **Cholesky** | $P \times P$ | $O(P^3)$ | [`regression.py`](../src/regression.py) |
| B4 | $\mathbf{c} = \mathbf{\Sigma}_M^{-1}\mathbf{t}$ = `cho_solve((L), target_vec)` | triangular solve | $P$ | $O(P^2)$ | [`regression.py:43`](../src/regression.py) |

$$
\text{Stage B} \;=\; O(P^3) \quad(\text{one Cholesky factorization + one triangular solve}).
$$

The factor $\mathbf{L}$ is persisted ([`regression.py:44`](../src/regression.py)) so
`variance.py` can reuse it for predictive variances.

---

## Complexity summary

| Stage | Dominant op | Complexity | Frequency |
|-------|-------------|-----------|-----------|
| A0 — $\mathbf{K}_{MM}$ Cholesky | Cholesky | $O(P^3)$ | once |
| **A1–A9 — Gram/target assembly** | per-molecule $P\times P$ update (chunked) | $O(P^2 \sum_i n_i) = O(P^2 N\bar n)$ | per molecule × $N$ |
| B3 — $\mathbf{\Sigma}_M$ Cholesky | Cholesky | $O(P^3)$ | once |
| B4 — weight solve | triangular solve | $O(P^2)$ | once |

Rewriting with $P \approx M\,n_{\text{atom}}$:

- Assembly: $O(P^2 N\bar n) = O\!\big(N \cdot A_{\text{mol}} \cdot M^2 \, n_{\text{atom}}^3\big)$
  — **linear in $N$**, **quadratic in $M$**.
- Solve: $O(P^3) = O(M^3\, n_{\text{atom}}^3)$ — **cubic in $M$**, independent of $N$.

---

## Which operation is the bottleneck?

There are two candidates, and which wins depends on the regime:

**1. Compute bottleneck — the Gram assembly, $O(P^2 \sum_i n_i)$ (Stage A).**
The streaming assembly dominates the total FLOP count. It is also the only part that 
scales with the dataset size $N$, so it is what grows as you add training data.

**2. Memory bottleneck — the dense $P \times P$ arrays (Stage A).**
Real difference compared to `regression_model = sagpr`. PITC keeps a dense $\mathbf{K}_{MM}$ factor resident
and accumulates each molecule's $\mathbf{G}_i$ (step A6); that per-molecule product used to be formed
as a **full dense $P \times P$ matrix**, but is now scattered in column chunks
([`gram_matrix.py:154`](../src/libs/gram_matrix.py)), leaving the resident dense $\mathbf{K}_{MM}$ as
the main excess over SA-GPR — which assembles the same dense Gram from small block updates with no large
temporaries (see the [contrast section](./training_complexity.md#contrast-sa-gpr-training-regression_model--sagpr)).

After chunking and $\mathbf{K}_{MM}$ sharing, each MPI rank holds essentially just the packed Gram
accumulator ($\tfrac{P^2}{2}\cdot 8$ B). The per-molecule
$\mathbf{G}_i = \mathbf{K}_{I_iM}^{\top}\mathbf{\Lambda}_i^{-1}\mathbf{K}_{I_iM}$ is accumulated in
column chunks bounded by `chunk_bytes` ([`gram_matrix.py:154`](../src/libs/gram_matrix.py)) rather
than as a full $P\times P$ temporary, and the dense $\mathbf{K}_{MM}$ Cholesky factor $\mathbf{L}_{MM}$
($P^2\cdot 8$ B) is shared **one copy per node** through MPI shared memory
([`_l_mm_shared`, gram_matrix.py](../src/libs/gram_matrix.py)) instead of built per rank. Together
these took the per-rank peak from ~$2.5\,P^2$ down to ~$0.5\,P^2$ (dz/512-ref: ~18.5 → ~4 GiB/rank).
The packed Gram accumulator is now the per-rank floor; capping ranks so $\text{ranks}\times 0.5\,P^2$
fits the node is the remaining constraint.

**In short:** the $O(P^3)$ Cholesky is the textbook GPR "cubic-in-inducing-points"
cost, but in this pipeline the **assembly of the dense PITC Gram matrix (Stage A) is the
real bottleneck** — it dominates both compute $O(M^2N)$ and memory O($P^2$).

---

## Contrast: SA-GPR training (`regression_model = sagpr`)

**The primary difference is memory, not asymptotic compute.** The SA-GPR Gram is *equally dense*:
the Coulomb metric $\mathbf{M}_i$ couples all $(\ell, q)$ blocks (in a QM7 example, 168 of its 225
metric TensorMap blocks have $\ell_1 \neq \ell_2$), so
$\mathbf{B} = \sum_i \mathbf{K}_{I_i M}^{\top}\,\mathbf{M}_i\,\mathbf{K}_{I_i M}$ fills the whole
$P \times P$ matrix just like the PITC Gram. Both assemblies are $O(P^2 \sum_i n_i)$ and both end in
an $O(P^3)$ solve — they are the **same complexity class**.

What actually makes SA-GPR lighter:

- **Memory (the main win).** `do_work_gram` ([`gram_matrix.py:76`](../src/libs/gram_matrix.py)) builds
  $\mathbf{B}$ from *small* per-reference-pair block updates (`einsum`s) scattered straight into the
  packed accumulator, and **never holds a dense $\mathbf{K}_{MM}$**. PITC's `do_work_gram_pitc`
  ([`gram_matrix.py:154`](../src/libs/gram_matrix.py)) now *also* scatters its contribution in column
  chunks (bounded by `chunk_bytes`), and its dense $\mathbf{K}_{MM}$ factor is shared one copy per
  node ([`_l_mm_shared`](../src/libs/gram_matrix.py)) rather than held per rank. Counting the
  $P^2$-sized arrays each rank holds:

  | | packed Gram | dense $\mathbf{K}_{MM}$ | per-mol dense $P{\times}P$ | total |
  |---|---|---|---|---|
  | SA-GPR                     | $0.5\,P^2$ | —                | —                | $0.5\,P^2$ |
  | DTC                        | $0.5\,P^2$ | —                | —                | $0.5\,P^2$ |
  | PITC (before)             | $0.5\,P^2$ | $1.0\,P^2$       | $1.0\,P^2$       | $2.5\,P^2$ |
  | PITC (chunked)            | $0.5\,P^2$ | $1.0\,P^2$       | bounded by chunk | $1.5\,P^2$ |
  | PITC (chunked + shared K) | $0.5\,P^2$ | shared, ~0/rank  | bounded by chunk | $0.5\,P^2$ |

  Chunking the per-molecule product removed a full $P\times P$ temporary (~$2.5\,P^2 \to 1.5\,P^2$),
  and node-sharing the read-only $\mathbf{K}_{MM}$ factor removed the last per-rank dense array
  (~$1.5\,P^2 \to 0.5\,P^2$), so PITC now matches SA-GPR's per-rank footprint — down from the
  ~20 GiB/rank that OOM-killed jobs. The packed Gram accumulator ($0.5\,P^2$) is the remaining
  per-rank floor.

- **Compute (a smaller, constant-factor win).** SA-GPR uses $\mathbf{M}_i$ directly, so it needs **no
  inversions**: no $\mathbf{S}_i^{-1}$, no $\mathbf{\Lambda}_i^{-1}$, no Nyström residual $\mathbf{D}_i$
  (which in PITC costs an extra per-molecule $O(P^2 n_i)$ $\mathbf{K}_{MM}$ solve, step A1), and no
  one-time $O(P^3)$ $\mathbf{K}_{MM}$ Cholesky during assembly. Same asymptotics, roughly half the
  constant.

- **Target** ([`target_vector.py:33`](../src/libs/target_vector.py)):
  $\mathbf{t} = \mathbf{K}_{NM}^{\top}\mathbf{w}$, one `einsum`.
- **Regularization + solve** ([`regression.py:47`](../src/regression.py)):
  $\mathbf{M} = \mathbf{B} + \text{reg}\cdot\mathbf{K}_{MM} + \text{jit}\cdot\mathbf{I}$, then
  $\mathbf{c} = $ `scipy.linalg.solve(M, t, assume_a='sym')` — a symmetric LDLᵀ solve, $O(P^3)$.

So SA-GPR and the GP models share the **same asymptotic cost** ($O(P^2 \sum_i n_i)$ assembly + $O(P^3)$
solve); SA-GPR is cheaper by a constant factor in compute and, decisively, by **~5× in dense-array
memory**. PITC pays that memory price to obtain a genuine GP posterior and predictive variances
(`variance.py`) *and* a better mean.

**DTC pays neither**: it runs this exact SA-GPR assembly, at this exact cost, and still gets
`variance.py` — because everything it needs beyond SA-GPR (the Cholesky factor, $\sigma_p^2$, the
$\mathbf{K}_{**}-\mathbf{Q}_{**}$ correction) is post-assembly and $O(P^3)$ at worst. What it does
*not* get is PITC's improved mean, which is precisely the $\mathbf{D}_i$ it discards, nor a
consistent joint GP. DTC is therefore the cheap way to put a principled error bar on an existing SA-GPR
model; PITC is the way to improve the model itself.

---

*Sources: [`src/get_matrices.py`](../src/get_matrices.py),
[`src/regression.py`](../src/regression.py),
[`src/libs/pitc_lib.py`](../src/libs/pitc_lib.py),
[`src/libs/gram_matrix.py`](../src/libs/gram_matrix.py),
[`src/libs/target_vector.py`](../src/libs/target_vector.py),
[`src/libs/kernels_lib.py`](../src/libs/kernels_lib.py),
[`src/libs/functions.py`](../src/libs/functions.py),
[`src/libs/config_paths.py`](../src/libs/config_paths.py).*
