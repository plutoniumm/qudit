# Primitives

## States
Qudit makes a state an almost first class citizen which allows us to easily create and manipulate quantum states. We can start by defining a basis, which for now is just the computational basis in some dimension. Consider a qutrit ($d=3$) system:

```python
from qudit import Basis

Ket = Basis(3)
```

And then declare a computational basis pure state via the basis. The following are all equivalent in creating the state $|012\rangle$:

```python
Psi = Ket("012")

Psi = Ket(0, 1, 2)

Psi = Ket(0) ^ Ket(1) ^ Ket(2)
```

We also have a `State` class which can be used to create more complex states with automatically normalized vectors. A result of this is that states are about as hard to declare in python as they are to write on paper. Consider

$$\begin{aligned}
|\psi\rangle &= \omega |0000\rangle + \omega^2|1010\rangle + \sqrt{3}i|2010\rangle \\
&+ |2200\rangle + (9i + 16)|1210\rangle + (\omega - \omega^2)|0022\rangle \\
&+ (\omega - 1)^2|2020\rangle + (e^{i\frac{\pi}{18}} + 6)|2221\rangle + |0112\rangle \\
&+ (5 + 9i)|1200\rangle + 0.67|1111\rangle + 9 e^{i\frac{\pi}{5}}|2222\rangle
\end{aligned}$$

::: code-group

```python [Example]
SV = State(
    w * Ket("0000")
    + w**2 * Ket("1010")
    + rt(3) * 1j * Ket("2010")
    + Ket("2200")
    + (9j + 16) * Ket("1210")
    + (w - w**2) * Ket("0022")
    + (w - 1) ** 2 * Ket("2020")
    + (e(1j * pi / 18) + 6) * Ket("2221")
    + Ket("0112")
    + (5 + 9j) * Ket("1200")
    + 0.67 * Ket("1111")
    + (9 * e(1j * pi / 16)) * Ket("2222")
)
```

```python [imports]
from qudit import State, Basis
import numpy as np

Unity = lambda d: np.exp(2j * np.pi / d)
e, rt = np.exp, np.sqrt
pi = np.pi

w, Ket = Unity(3), Basis(4)
```

:::

### Properties of states
States contain several properties, and methods for information and manipulation

| Property | Description |
| --- | --- |
| `d` | dimension of state (e.g. 2 for qubits, 3 for qutrits) |
| `H` | Hermitian conjugate via `.conj().T` |
| `trace` | Return the trace of the state (if density matrix) else return norm squared (if pure state) |
| `isDensity` | if state is a density matrix (True) else pure state (False) |
| `isPure()` | if state is pure (True) else density matrix (False) |
| `norm()` | Return the state normalized to unit norm (if not) |
| `density()` | Return the density matrix of the state |
| `proj()` | projector $\|\psi\rangle\langle\psi\|$ if pure state else $\rho$ |
| `oproj()` | perpendicular projector $I - \|\psi\rangle\langle\psi\|$ if pure state else $I - \rho$ |


## Gates
Gates in `qudit` are operators (matrices) that act on states and states. With the same preliminaries as above;

### Single-qudit examples
We use the operations
- `@` to apply matmult of an operator on a state, and similarly
- `^` to build tensor products of states and gates.

For example, for a qubit system:

::: code-group

```python [Example]
Ket = Basis(2)
G = Gategen(2)

psi = Ket("0")

# Apply a Hadamard
psi = G.H @ psi
print(psi)

# Apply a rotation
psi = G.RY(np.pi/2) @ psi
print(psi) # results in |1>
```


```python [imports]
from qudit import Basis
from qudit.circuit import Gategen
```

:::

The above system creates $RY(\pi/2)|+\rangle = |1\rangle$.

### Three-qudit GHZ
Similarly we can start with $|000\rangle$ and apply a Hadamard on the first qubit followed by two CNOTs to create a three-qubit GHZ state:

$|\text{GHZ}\rangle = \frac{|000\rangle + |111\rangle}{\sqrt{2}}$.

::: code-group

```python [Example]
Ket = Basis(2)
G = Gategen(2)

psi = Ket("000")

HII = G.H ^ G.I ^ G.I
CX1 = G.CX ^ G.I
CX2 = G.I ^ G.CX

psi = CX2 @ (CX1 @ (HII @ psi))
print(psi)
```


```python [imports]
from qudit import Basis
from qudit.circuit import Gategen
```

:::

## Gategen Generics
Gates are made up of two interoperable classes `Unitary` and `Gate`, both of which interop with each other, states, and tensors. The following gates are available in `Gategen` for a given dimension `d` in the circuit-less form:

| Name | Symbol | Description |
| --- | --- | --- |
| Identity | $I$ | Identity gate |
| Hadamard | $H$ | Generalized Hadamard gate |
| X | $X$ | Generalized Pauli X gate |
| Y | $Y$ | Generalized Pauli Y gate |
| Z | $Z$ | Generalized Pauli Z gate |
| RX($\theta$) | $RX(\theta)$ | X rotation axis by $\theta$ |
| RY($\theta$) | $RY(\theta)$ | Y rotation axis by $\theta$ |
| RZ($\theta$) | $RZ(\theta)$ | Z rotation axis by $\theta$ |
| CX | $CX$ | CX gate |
| CSUM | $CSUM$ | Generalized CX gate, or the CSUM gate such that $CSUM\|i\rangle\|j\rangle = \|i\rangle\|i+j \mod d\rangle$ |
| SWAP | $SWAP$ | Generalized SWAP gate such that $SWAP\|i\rangle\|j\rangle = \|j\rangle\|i\rangle$ |

## Gategen Special Gates
In addition to these gates there are two special gates called `CU` or `Controlled-U` and `GMR` or `GellMann Rotation` used as

- `CU` is a generalization of the CX gate where the target operation can be any unitary, and the control can be on any state. For example, `CU(control=1, target=G.H)` applies a Hadamard on the target qudit if the control qudit is in state $|1\rangle$.
- `GMR` is a generalization of the RX, RY, RZ gates where the rotation can be on any axis defined by two states. For example, `GMR(0, 2, np.pi/2)` applies a $\pi/2$ rotation on the axis defined by $|0\rangle$ and $|2\rangle$.

## Common States

`qudit.tools` provides standard multi-partite states. All return normalized `State` objects.

| Function | Description |
| --- | --- |
| `GHZ(n, d)` | $n$-partite qudit GHZ: $\frac{1}{\sqrt{d}}\sum_{i=0}^{d-1}\|i\rangle^{\otimes n}$ |
| `W(n)` | $n$-qubit W state: $\frac{1}{\sqrt{n}}\sum_i \|0\cdots 1_i\cdots 0\rangle$ |
| `NOON(n, theta)` | Two-mode NOON state: $\|n,0\rangle + e^{in\theta}\|0,n\rangle$ in dimension $n+1$ |
| `Dicke(n, k)` | $n$-qubit Dicke state with $k$ zeros and $n-k$ ones |
| `Coherent(N, alpha)` | Truncated coherent state in $(N+1)$-dimensional Fock space |

::: code-group

```python [Example]
print(GHZ(3, 2))    # (|000> + |111>) / sqrt(2)
print(W(3))         # (|100> + |010> + |001>) / sqrt(3)
print(NOON(3, 0))   # (|3,0> + |0,3>) / sqrt(2) in d=4
print(Dicke(4, 1))  # uniform superposition of weight-1 states
print(Coherent(5, alpha=1.0))
```

```python [imports]
from qudit.tools.states import GHZ, W, NOON, Dicke, Coherent
```

:::

## Random

`qudit.random` samples from the Haar measure on $U(n)$:

| Function | Description |
| --- | --- |
| `random_unitary(n)` | Haar-random $n\times n$ unitary via complex Ginibre QR |
| `random_state(n)` | Haar-random pure state $\|\psi\rangle \in \mathbb{C}^n$ |

```python
from qudit import random_unitary, random_state

U = random_unitary(4)   # 4x4 unitary matrix (np.ndarray)
psi = random_state(4)   # random 4-component statevector
```

## Separability and decomposition (`Space`, `PPT`)

`qudit.tools` provides tools for analyzing bipartite states:

| Function/Method | Description |
| --- | --- |
| `PPT(rho, sub)` | Peres–Horodeccy partial-transpose test: `True` if $\rho^{T_B}\succeq 0$ |
| `Space.gramSchmidt(vectors)` | Gram-Schmidt orthonormalization |
| `Space.schmidtDecompose(state)` | Schmidt decomposition; returns list of $(\lambda_k, \|u_k\rangle, \|v_k\rangle)$ |
| `Space.schmidtRank(mat)` | Schmidt rank (matrix rank) of a bipartite coefficient matrix |

::: code-group

```python [Example]
# Bell state: maximally entangled, Schmidt rank 2
psi = (np.kron([1, 0], [1, 0]) + np.kron([0, 1], [0, 1])) / np.sqrt(2)
mat = psi.reshape(2, 2)  # bipartite coefficient matrix

decomp = Space.schmidtDecompose(mat)
print(Space.schmidtRank(mat))  # 2

rho = np.outer(psi, psi.conj())
print(PPT(rho, 2))  # False  (entangled states fail PPT)
```

```python [imports]
from qudit.tools.tests import Space, PPT
import numpy as np
```

:::

> [!NOTE]
> `PPT` currently overrides the `sub` argument to `3` internally; pass the actual block size you intend and verify against the matrix dimensions.