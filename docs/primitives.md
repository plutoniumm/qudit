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