<img src="./docs/icon.svg" width="75" height="75" align="right">

### Qudit
Simulations for Qudit systems

[![PyPI version](https://badge.fury.io/py/qudit.svg)](https://pypi.org/project/qudit/)

## Usage
After `pip install qudit`, you can use the package as follows:


### Usage
Two qubit example: bell state

```python
from qudit import *
D = Gategen(2) # create gateset for 2-dits
Ket = Basis(2)
# p = |00><00|
p = Ket(0, 0).density()

rho = D.CX @ D.H ^ D.I @ p # p - H x I - CX
print(rho)

# Tr(|11><11| p)
prob = Ket("11").density().dot(rho)
print(np.trace(prob)) # 0.5
```

Three Qutrit example: 3-GHZ state = $\frac{1}{\sqrt{3}}(|000\rangle + |111\rangle + |222\rangle)$
```python
from qudit import *
Ket = Basis(3)

P000 = Ket(0) ^ Ket(0) ^ Ket(0)
P111 = Ket(1, 1, 1)
P222 = Ket("222")

# |GHZ> = 1/sqrt(3) * (|000> + |111> + |222>)
GHZ = State(P000 + P111 + P222)

print(GHZ.density()) # |GHZ><GHZ|
```

### Caveats
- I don't plan to support Variational circuits anytime soon. For that probably use `cirq`, or hand-write your circuits.


### Todos
- Remove `O is None` checks from `Gate` and `VarGate`


**Done**:
- Define statevector
- Gates: Rot(GellMann), H, CU, CX, XYZ, TSP
- Circuit: Simulation, Drawing
- Random: States, Gates
- GramSchmidt
- Fidelity: Fid, EntFid, Channel Fid
- Negativity
- Standard States: GHZ, W, Cohorent, Dicke

<!-- **Almost Done**: -->

**Not Done**:
- Partial Trace
- Gates: QFT
- Noise: Kraus, Choi
- Stabiliser → States
- States → Stabiliser
- Discord
- Entropy
