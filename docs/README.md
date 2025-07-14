<img src="./icon.svg" width="125" height="125" align="right" style="position:relative;top:60px;">

# Qudit
Sparse Matrix simulations for qudit systems. To make qudit machine learning, qudit error correction, and qudit circuit simulation easier.

[![PyPI version](https://badge.fury.io/py/qudit.svg)](https://pypi.org/project/qudit/)

# Usage
`pip install qudit`

## Quickstart
In most cases it should not matter if you mix and match `numpy` with `qudit` since most abstractions are built on top of `numpy` arrays. The following is two examples to do the same thing, one using the `Circuit` class and the other manually using the matrices.

```python
from qudit import Circuit
k00 = np.array([1, 0, 0, 0])
k00 = np.outer(k00, k00)  # |00><00|

C = Circuit(2, dim=2) # 2 quDits with dim 2
G = C.gates

C.gate(G.H, dits=[0])
C.gate(G.CX, dits=[0, 1])

U = C.solve()
U @ k00 @ U.T # Tr = 1
```

This same thing can be done with the `Circuit` class:

```python
from qudit import Gategen, Basis

D = Gategen(2)
Ket = Basis(2)

k00 = Ket(0, 0).density() # |00><00|

rho = D.CX @ (D.I ^ D.H)

rho @ k00 @ rho.H # Tr = 1
```