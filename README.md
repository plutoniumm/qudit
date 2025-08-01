<img src="https://raw.githubusercontent.com/plutoniumm/qudit/v2/docs/_static/icon.svg" alt="icon" width="125" height="125" align="right" name="icon"/>

# Qudit

<!-- 147 -->

Sparse Matrix simulations for qudit systems. To make qudit machine learning, qudit error correction, and qudit circuit simulation easier. Qudit is made fully around `numpy` and `scipy` to make it easy to mix and match tools without worrying about type errors.

[![PyPI version](https://badge.fury.io/py/qudit.svg)](https://pypi.org/project/qudit/)

```bash
pip install qudit
```

## Quickstart

In most cases it should not matter if you mix and match `numpy` with `qudit` since most abstractions are built on top of `numpy` arrays. The following is two examples to do the same thing, one using the `Circuit` class and the other manually using the matrices.

**Using the `Circuit` class:**

```python
from qudit import Circuit
import numpy as np

k00 = np.array([1, 0, 0, 0])
k00 = np.outer(k00, k00)  # |00⟩⟨00|

C = Circuit(2, dim=2)  # 2 quDits with dim 2
G = C.gates

C.gate(G.H, dits=[0])
C.gate(G.CX, dits=[0, 1])

U = C.run()
U @ k00 @ U.T  # Tr = 1
```

**Manual matrix version:**

```python
from qudit import Gategen, Basis

D = Gategen(2)
Ket = Basis(2)

k00 = Ket(0, 0).density()  # |00><00|

rho = D.CX @ (D.I ^ D.H)

rho @ k00 @ rho.H  # Tr = 1
```

## Not Done

* Partial Trace
* Gates: QFT
* Noise: Kraus, Choi
* States → Stabiliser
* Discord