# Supplementary computational verification

Companion code for the manuscript
*No Universal Process Currying: The Intercept Principle, Environmental
Currying, and Finite-Dimensional Obstructions to Exact Process Storage,
Evaluation, and Recovery.*

## Contents

- `verify_abaee_currying.py` — a self-contained verification suite (73 checks)
  certifying the paper's arithmetic, symbolic, and numerical claims.

## What is verified

The script is organized into labelled groups, each certifying a specific
class of statements in the paper:

| Group | Certifies |
|-------|-----------|
| A | Arithmetic/dimension obstructions: $e^4-e^2+1$ never a perfect square; instrument grade-intercept mismatch; FinStoch vertex-count inequality; affine-dimension identities. |
| B | Automatic convex enrichment via the control-bit construction. |
| C | No-programming / no finite-memory universal evaluator (SDP feasibility). |
| D | Diamond-norm rank-defect bound: sharp constant $c(e)=e/2$; interior-ball radius $2/e^2$; certified diamond-norm SDP (primal = dual); exhaustive rank-drop sweeps. |
| E | Petz recoverability: the four equivalent conditions flip together across a channel zoo. |
| F | Higher-order transfer: necessity of the reflection hypothesis (CPM counterexample). |
| H | Symbolic Kraus–Gram algebra (exact, general $n$ and $\dim H$). |
| I | Numerical Kraus–Gram / recovery (floating point). |
| J | Intercept and graded-intercept engine. |
| K | Exact instrument constant $c_n(e)=e/2$ (independent of $n$); instrument in-radius $2/(ne^2)$ with analytic witness. |
| L | Approximate finite-memory floor $\dim(M)\ge e$ (packing + data-processing contraction). |
| M | Grade-forgetful total-map quotient $\cong\mathbf{Chan}$. |
| N | Deterministic superchannel affine dimension; reflection failure by wire-bending. |
| O | Multiset grade-forgetful quotient: split-monomorphism structure, one-outcome exclusion, non-surjectivity, and dimension bounds. |

## Requirements

- Python 3.9+
- `numpy`, `scipy`, `sympy`
- `cvxpy` (for the semidefinite-programming checks; SCS solver)

The semidefinite checks (diamond norms, no-programming, instrument constants)
require `cvxpy`. Without it, those checks are skipped and the remaining
arithmetic/symbolic checks still run.

## Running

```
pip install numpy scipy sympy cvxpy
python3 verify_abaee_currying.py
```

Expected output ends with a summary line reporting all checks passed. Each
check prints `[PASS]` or `[FAIL]` with a description; a `FAIL` would list the
offending check.

## Notes on the numerics

- Diamond norms are computed as certified semidefinite programs in the
  induced-trace-norm convention (Watrous primal, cross-checked against the
  matched dual for strong-duality agreement), so reported values are
  certified from both sides.
- Suprema over process spaces are established by matching analytic witnesses,
  not by random search alone; random search is used only for corroboration and
  as regression guards against known non-extremal directions.
