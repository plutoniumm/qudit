<img src="https://raw.githubusercontent.com/plutoniumm/qudit/refs/heads/main/docs/_static/icon.png" alt="icon" width="125" height="125" align="right" name="icon"/>

### `qudit`

High performance simulations for qudit systems. To make qudit machine learning, qudit error correction, and qudit circuit simulation easier. Qudit is made fully around `numpy` and `pytorch` to make it easy to mix and match tools without worrying about type errors.

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

C = Circuit(2, dim=2)  # 2 qBits with d=2
G = C.gates[2]

C.gate(G.H, dits=[0])
C.gate(G.CX, dits=[0, 1])

ket0 = np.zeros(2**2)
ket0[0] = 1.0  # |00>

print(C(ket0))  # [1. 0. 0. 1.]/rt2
```
