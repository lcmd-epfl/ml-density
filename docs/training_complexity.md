# Training: matrix operations and computational complexity

This note walks through **every linear-algebra operation performed during training**
of the SA-GPR electron-density model, gives each one a complexity, and identifies
the bottleneck.

Training is split across two scripts:

1. **`src/get_matrices.py -b`** — assembles the **Gram matrix** and the **target vector**
   (one streaming pass over the training molecules).
2. **`src/regression.py`** — solves one symmetric positive-definite linear system for the
   **regression weights**.

The code has two training modes, selected by the config flag `full_gpr`
([`config_paths.py:38`](../src/libs/config_paths.py)):

- `full_gpr = True` → **PITC** (Partially Independent Training Conditional) — the
  "full-GPR" path, documented in detail below.
- `full_gpr = False` → **SoR** (Subset of Regressors) — the default sparse path,
  summarized for contrast at the end.

Both are *sparse / inducing-point* methods over `M` reference environments; neither ever
forms the dense `N × N` kernel over all training atoms.

---

## Notation

| Symbol | Meaning | Code name / location |
|--------|---------|----------------------|
| $N$ | number of training molecules | loop bound over `ntrains` ([`get_matrices.py:26`](../src/get_matrices.py)) |
| $M$ | number of reference (inducing) environments | `o.M` ([`config_paths.py:32`](../src/libs/config_paths.py)) |
| $n_i$ | number of density-fitting AO coefficients for molecule $i$ | `nao_i` ([`pitc_lib.py:82`](../src/libs/pitc_lib.py)) |
| $\bar n$ | average $n_i$; $\sum_i n_i$ = total training AO coefficients | — |
| $P$ | **problem dimensionality** = total AO coefficients over the $M$ references = size of the weight vector and of the Gram matrix | `nao_ref = basis.nao_for_mol(ref_elements)` ([`regression.py:24`](../src/regression.py), [`gram_matrix.py:245`](../src/libs/gram_matrix.py)) |
| $\eta$ | PITC noise scale (per-molecule precision) | `o.reg` ([`config_paths.py:36`](../src/libs/config_paths.py)) |

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
  often ill-conditioned (κ ≈ 10⁷), hence diagonal jitter ([`pitc_lib.py:76`](../src/libs/pitc_lib.py)).
- $\mathbf{w}_i$ — metric-projected density-fitting coefficients (the fit target),
  $(n_i,)$.

Kernels use a **ζ = 2** symmetry-adapted (λ-SOAP) nonlinearity: each $\lambda > 0$ block is
multiplied by the scalar $\lambda = 0$ kernel ([`kernels_lib.py:147`](../src/libs/kernels_lib.py)).

---

## Full-GPR (PITC) training

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
\boxed{\;\mathbf{t} \;=\; \sum_{i=1}^{N} \mathbf{K}_{M I_i}\,\mathbf{\Lambda}_i^{-1}\,\mathbf{S}_i^{-1}\,\mathbf{w}_i\;}
$$

and the weights are the solution of

$$
\boxed{\;\mathbf{\Sigma}_M\,\mathbf{c} \;=\; \mathbf{t} \;\;\Longrightarrow\;\; \mathbf{c} = \mathbf{\Sigma}_M^{-1}\,\mathbf{t}.\;}
$$

### Stage A — assemble $\mathbf{\Sigma}_M$ and $\mathbf{t}$  (`get_matrices.py -b`)

**Setup (once).** Densify $\mathbf{K}_{MM}$, add jitter, factorize:

| # | Operation | Kind | Shape | Complexity | Code |
|---|-----------|------|-------|-----------|------|
| A0 | $\mathbf{L}_{MM} = \operatorname{chol}(\mathbf{K}_{MM} + \text{jit}\cdot\mathbf{I})$ | Cholesky | $P \times P$ | $O(P^3)$ | [`pitc_lib.py:33`](../src/libs/pitc_lib.py) |

**Per training molecule $i$** (`molecule_lambda_inv_knm`, [`pitc_lib.py:62`](../src/libs/pitc_lib.py)):

| # | Operation | Kind | Shape | Complexity | Code |
|---|-----------|------|-------|-----------|------|
| A1 | $\mathbf{y}_i = \mathbf{K}_{MM}^{-1}\mathbf{K}_{M I_i}$ = `cho_solve((L_MM), k_nm_i.T)` | triangular solve, $n_i$ RHS | $P \times n_i$ | $O(P^2 n_i)$ | [`pitc_lib.py:98`](../src/libs/pitc_lib.py) |
| A2 | $\mathbf{D}_i = \mathbf{K}_{I_i I_i} - \mathbf{K}_{I_i M}\mathbf{y}_i$ | matrix–matrix | $n_i \times n_i$ | $O(n_i^2 P)$ | [`pitc_lib.py:99`](../src/libs/pitc_lib.py) |
| A3 | $\mathbf{S}_i^{-1}$ = `cho_factor` + `cho_solve(I)` | Cholesky + solve → explicit inverse | $n_i \times n_i$ | $O(n_i^3)$ | [`pitc_lib.py:101`](../src/libs/pitc_lib.py) |
| A4 | $\mathbf{\Lambda}_i = \mathbf{D}_i + \eta\,\mathbf{S}_i^{-1}$ | add | $n_i \times n_i$ | $O(n_i^2)$ | [`pitc_lib.py:102`](../src/libs/pitc_lib.py) |
| A5 | $\mathbf{\Lambda}_i^{-1}$ = `np.linalg.inv` | dense inverse (LU) | $n_i \times n_i$ | $O(n_i^3)$ | [`pitc_lib.py:103`](../src/libs/pitc_lib.py) |
| A6 | $\mathbf{G}_i = \mathbf{K}_{I_i M}^{\top}\,\mathbf{\Lambda}_i^{-1}\,\mathbf{K}_{I_i M}$ | matrix products in **column chunks** (no full $P\times P$ temporary) | temp $\le$ `chunk_bytes` | $O(n_i^2 P) + O(P^2 n_i)$ | [`gram_matrix.py:154`](../src/libs/gram_matrix.py) |
| A7 | scatter each chunk's columns into packed Gram | accumulate | $P(P{+}1)/2$ | $O(P^2)$ | [`gram_matrix.py:157`](../src/libs/gram_matrix.py) |
| A8 | $\mathbf{S}_i^{-1}\mathbf{w}_i$ = `cho_solve` | triangular solve, matrix–vector | $n_i$ | $O(n_i^2)$ | [`target_vector.py:66`](../src/libs/target_vector.py) |
| A9 | $\mathbf{t}_i = \mathbf{K}_{I_i M}^{\top}(\mathbf{\Lambda}_i^{-1}\mathbf{S}_i^{-1}\mathbf{w}_i)$ | matrix–vector | $P$ | $O(P n_i)$ | [`target_vector.py:67`](../src/libs/target_vector.py) |

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

### Stage B — the training solve (`regression.py`, `full_gpr` branch)

| # | Operation | Kind | Shape | Complexity | Code |
|---|-----------|------|-------|-----------|------|
| B0 | densify $\mathbf{K}_{MM}$ (`kmm_cholesky`) | densify | $P \times P$ | $O(P^2)$ | [`regression.py:33`](../src/regression.py) |
| B1 | `unravel_tril` — unpack packed Gram → dense | unpack | $P \times P$ | $O(P^2)$ | [`regression.py:38`](../src/regression.py) |
| B2 | $\mathbf{\Sigma}_M = \text{Gram} + \mathbf{K}_{MM}$ | add | $P \times P$ | $O(P^2)$ | [`regression.py:39`](../src/regression.py) |
| B3 | $\mathbf{L} = \operatorname{chol}(\mathbf{\Sigma}_M)$ (`robust_cholesky`, retries with ×10 jitter) | **Cholesky** | $P \times P$ | $O(P^3)$ | [`regression.py:40`](../src/regression.py) |
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
Real difference compared to `full_gpr = False`. PITC keeps a dense $\mathbf{K}_{MM}$ factor resident
and accumulates each molecule's $\mathbf{G}_i$ (step A6); that per-molecule product used to be formed
as a **full dense $P \times P$ matrix**, but is now scattered in column chunks
([`gram_matrix.py:154`](../src/libs/gram_matrix.py)), leaving the resident dense $\mathbf{K}_{MM}$ as
the main excess over SoR — which assembles the same dense Gram from small block updates with no large
temporaries (see the [contrast section](./training_complexity.md#contrast-sor-training-full_gpr--false)).

Each MPI rank holds the packed Gram accumulator ($\tfrac{P^2}{2}\cdot 8$ B) and the dense
$\mathbf{K}_{MM}$ behind $\mathbf{L}_{MM}$ ($P^2\cdot 8$ B). The per-molecule
$\mathbf{G}_i = \mathbf{K}_{I_iM}^{\top}\mathbf{\Lambda}_i^{-1}\mathbf{K}_{I_iM}$ used to add a
*third* dense $P\times P$ array, but is now accumulated in column chunks bounded by `chunk_bytes`
([`gram_matrix.py:154`](../src/libs/gram_matrix.py)), so the per-rank peak is ~$1.5\,P^2$ (down
from ~$2.5\,P^2$). This is still $O(P^2)$ per rank and the practical limiter; ranks must be capped
so total memory fits the node. A further drop to ~$0.5\,P^2$ is possible by node-sharing the
read-only $\mathbf{L}_{MM}$ (see the contrast section).

**In short:** the $O(P^3)$ Cholesky is the textbook GPR "cubic-in-inducing-points"
cost, but in this pipeline the **assembly of the dense PITC Gram matrix (Stage A) is the
real bottleneck** — it dominates both compute $O(M^2N)$ and memory O($P^2$).

---

## Contrast: SoR training (`full_gpr = False`)

**The primary difference is memory, not asymptotic compute.** The SoR Gram is *equally dense*:
the Coulomb metric $\mathbf{M}_i$ couples all $(\ell, q)$ blocks (in a QM7 example, 168 of its 225
metric TensorMap blocks have $\ell_1 \neq \ell_2$), so
$\mathbf{B} = \sum_i \mathbf{K}_{I_i M}^{\top}\,\mathbf{M}_i\,\mathbf{K}_{I_i M}$ fills the whole
$P \times P$ matrix just like the PITC Gram. Both assemblies are $O(P^2 \sum_i n_i)$ and both end in
an $O(P^3)$ solve — they are the **same complexity class**.

What actually makes SoR lighter:

- **Memory (the main win).** `do_work_gram` ([`gram_matrix.py:76`](../src/libs/gram_matrix.py)) builds
  $\mathbf{B}$ from *small* per-reference-pair block updates (`einsum`s) scattered straight into the
  packed accumulator, and **never holds a dense $\mathbf{K}_{MM}$**. PITC's `do_work_gram_pitc`
  ([`gram_matrix.py:154`](../src/libs/gram_matrix.py)) now *also* scatters its contribution in column
  chunks (bounded by `chunk_bytes`), but still keeps the dense $\mathbf{K}_{MM}$ factor resident.
  Counting the $P^2$-sized arrays each rank holds:

  | | packed Gram | dense $\mathbf{K}_{MM}$ | per-mol dense $P{\times}P$ | total |
  |---|---|---|---|---|
  | SoR            | $0.5\,P^2$ | —          | —                  | $0.5\,P^2$ |
  | PITC (before)  | $0.5\,P^2$ | $1.0\,P^2$ | $1.0\,P^2$         | $2.5\,P^2$ |
  | PITC (chunked) | $0.5\,P^2$ | $1.0\,P^2$ | bounded by chunk   | $1.5\,P^2$ |

  Chunking the per-molecule product removed a full $P\times P$ temporary, cutting PITC's peak from
  ~$2.5\,P^2$ to ~$1.5\,P^2$ (~3× SoR) — this is what the ~20 GiB/rank requirement was about.
  Node-sharing the read-only $\mathbf{K}_{MM}$ factor across ranks would drop it further to
  ~$0.5\,P^2$, matching SoR.

- **Compute (a smaller, constant-factor win).** SoR uses $\mathbf{M}_i$ directly, so it needs **no
  inversions**: no $\mathbf{S}_i^{-1}$, no $\mathbf{\Lambda}_i^{-1}$, no Nyström residual $\mathbf{D}_i$
  (which in PITC costs an extra per-molecule $O(P^2 n_i)$ $\mathbf{K}_{MM}$ solve, step A1), and no
  one-time $O(P^3)$ $\mathbf{K}_{MM}$ Cholesky during assembly. Same asymptotics, roughly half the
  constant.

- **Target** ([`target_vector.py:33`](../src/libs/target_vector.py)):
  $\mathbf{t} = \mathbf{K}_{NM}^{\top}\mathbf{w}$, one `einsum`.
- **Regularization + solve** ([`regression.py:47`](../src/regression.py)):
  $\mathbf{M} = \mathbf{B} + \text{reg}\cdot\mathbf{K}_{MM} + \text{jit}\cdot\mathbf{I}$, then
  $\mathbf{c} = $ `scipy.linalg.solve(M, t, assume_a='sym')` — a symmetric LDLᵀ solve, $O(P^3)$.

So SoR and full-GPR share the **same asymptotic cost** ($O(P^2 \sum_i n_i)$ assembly + $O(P^3)$
solve); SoR is cheaper by a constant factor in compute and, decisively, by **~5× in dense-array
memory**. Full-GPR pays that memory price to obtain a valid PITC posterior and predictive variances
(`variance.py`).

---

*Sources: [`src/get_matrices.py`](../src/get_matrices.py),
[`src/regression.py`](../src/regression.py),
[`src/libs/pitc_lib.py`](../src/libs/pitc_lib.py),
[`src/libs/gram_matrix.py`](../src/libs/gram_matrix.py),
[`src/libs/target_vector.py`](../src/libs/target_vector.py),
[`src/libs/kernels_lib.py`](../src/libs/kernels_lib.py),
[`src/libs/functions.py`](../src/libs/functions.py),
[`src/libs/config_paths.py`](../src/libs/config_paths.py).*
