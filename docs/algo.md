# Algorithms

`qudit.algo` provides higher-level tools:
- `Statiliser` finds the stabilized subspace of a set of Pauli operators
- `qudit.tools.entanglement` computes the entanglement rank of a bipartite subspace.

## Stabilizer codes (`Statiliser`)

A stabilizer code $\mathcal{S}$ is a subspace defined as the simultaneous $+1$ eigenspace of a commuting set of Pauli operators $\{g_1, \ldots, g_k\}$:

$$|\psi\rangle \in \mathcal{C} \iff g_j|\psi\rangle = |\psi\rangle \quad \forall j$$

For $n$ qudits of local dimension $d$ with $k$ independent generators, the codespace has dimension $d^{n-k}$.

`Statiliser` finds this subspace numerically via gradient-free optimization followed by Gram–Schmidt orthonormalization.

### Creating a Statiliser

Stabilizers are given as strings over `{I, X, Y, Z}`. `d` (optional) sets the local qudit dimension; for `d>2`, `X` and `Z` are interpreted as the clock/shift operators from `Gategen(d)`.

::: code-group

```python [Qubit (d=2)]
stabs = ["ZZZII", "IIZZZ", "XIXXI", "IXXIX"]
s = Statiliser(stabs)

print(s.num_states)  # 2  (= 2^(5-4))
print(s.sz)          # 5
```

```python [Qutrit (d=3)]
s = Statiliser(["ZI", "IZ"], d=3)

print(s.num_states)  # 1  (= 3^(2-2))
print(s.sz)          # 2
```

```python [imports]
from qudit.algo import Statiliser
```

:::

| Property | Description |
| --- | --- |
| `sz` | Number of qudits ($\log_d$ of operator dimension) |
| `num_states` | Codespace dimension $d^{n-k}$ |
| `stabilisers` | List of stabilizer operator tensors |
| `basis` | Orthonormal basis after `generate()` (initially `None`) |

### Generating the codespace basis

::: code-group

```python [Example]
stabs = ["ZZZII", "IIZZZ", "XIXXI", "IXXIX"]

basis = Statiliser(stabs).generate(minimal=False)
# basis: Tensor of shape (num_states, 2**sz)

# Verify: each codeword is a +1 eigenstate
for state in basis:
    state_c = state.to(pt.complex64)
    for stab_str in stabs:
        G = S(stab_str).to(pt.complex64)
        diff = pt.norm(G @ state_c - state_c).item()
        print(f"{stab_str}: ||g|ψ> - |ψ>|| = {diff:.2e}")
```

```python [imports]
from qudit.algo import Statiliser
from qudit.algo.statiliser import S
import torch as pt
```

:::

`generate` takes two key options:

| Argument | Default | Description |
| --- | --- | --- |
| `mode` | `"real"` | `"real"` for real-valued search; `"complex"` for complex search |
| `minimal` | `True` | If `True`, adds L1+L2 regularization (prefers sparse solutions) |
| `tol` | `1e-6` | Optimizer convergence tolerance |

> [!NOTE]
> `minimal=True` includes L1 regularization which can prevent finding exact $+1$ eigenstates. For eigenstate verification tests, use `minimal=False` with a fixed seed (`torch.manual_seed`).

### Pauli string helper

`S(string)` converts a Pauli string to its full operator tensor (via Kronecker product):

```python
from qudit.algo.statiliser import S

G = S("ZXI")  # Z ⊗ X ⊗ I as a torch.Tensor
```

## Entanglement rank

The entanglement rank of a bipartite subspace measures how entangled a given space is, beyond what can be expressed with product states. It is computed numerically as the minimum of a quadratic projector loss over a parameterized family.

### Key functions

| Function | Description |
| --- | --- |
| `Perp(basis)` | Orthogonal projector onto the complement of `span(basis)` |
| `Loss(F, projector)` | Quadratic form $F^\dagger \Pi F$ (objective value) |
| `rank(f, D, r)` | Multi-start minimization returning smallest objective found |

### Computing entanglement rank

- define a subspace
- build its complement projector
- minimize the objective over product states.

::: code-group

```python [Example]
THETA = 0.75
Bits, Trits = Basis(2), Basis(3)

def Psi(i):
    A = Bits(0) ^ Trits(i)
    B = Bits(1) ^ Trits(i + 1)

    return A * np.cos(THETA) + B * np.sin(THETA)

perp = Perp([Psi(0), Psi(1)])

toCplx = np.array([1, 1j])
def system(X):
    qbit  = State(X[1:5].reshape(2, 2).dot(toCplx))
    qtrit = State(X[5:11].reshape(3, 2).dot(toCplx))
    phi_rx = (X[0] * (qbit ^ qtrit)).norm()

    return Loss(phi_rx, perp)

print(rank(system, D=5, r=2, tries=2))  # ≈ 0.2481
```

```python [imports]
from qudit.tools.entanglement import Perp, Loss, rank
from qudit import Basis, State
import numpy as np
```

:::

### `rank(f, D, r)` arguments

| Argument | Description |
| --- | --- |
| `f` | Objective function `f(x) -> float` |
| `D` | Parameter vector size hint (used to compute `size = (2D+1)(r-1)`) |
| `r` | Rank of the parameterized family |
| `tries` | Number of random restarts (default `1`) |
| `method` | Scipy optimizer method (default `"Powell"`) |
| `tol` | Convergence tolerance (default `~1e-10`) |

> [!TIP]
> Use `tries > 1` to reduce sensitivity to bad initial conditions;  `rank` returns the minimum over all restarts.
