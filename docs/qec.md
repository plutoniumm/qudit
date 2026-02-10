# Quantum Error Correction (QEC)

`qudit` includes a small set of QEC-oriented primitives for:

- constructing noise processes (channels),
- computing recovery maps,
- and evaluating how well a code performs under noise + recovery.

The main entry points are:

- `qudit.noise.Process` for building noise channels,
- `qudit.noise.Recovery` for building recovery channels,
- `qudit.tools.Fidelity` for computing performance metrics.

This page shows the typical workflow used in the test suite (see `tests/ECC.py`):

1) define a code subspace,
2) define a noise process acting on the physical system,
3) compute a recovery (Petz or Leung),
4) evaluate entanglement fidelity.

## Code representation

A (subspace) code is represented as an array of codewords, one per logical basis state.
In the tests, the code is a 2-dimensional codespace embedded in a 16-dimensional physical Hilbert space (so: `k=2`, `n=16`).

- Shape: `(k, n)`
- Each row is a codeword statevector.
- Codewords should be normalized.

Example (from `tests/ECC.py`):

```python
import numpy as np
from numpy import linalg as LA

code = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0.0],
    ],
    dtype=np.complex64,
)

# normalize each codeword
code /= LA.norm(code, axis=1)[:, None]
```

## Noise processes (`Process`)

Noise is constructed using `qudit.noise.Process`. A `Process` is a channel acting on the physical system.

A common pattern is to build a parameterized channel (e.g. a damping channel) and then derive recovery operations from it.

### Example: generalized amplitude damping

```python
from qudit.noise import Process

# Example from tests:
# local dimension d=2, number of physical particles/wires n=4
# (so total physical state dimension is 2**4 = 16)
ops = Process.GAD(2, 4, Y=0.01, p=0.001)
```

> Notes
> - The exact meaning of parameters like `Y` and `p` is channel-specific.
> - The first two positional arguments typically determine the local dimension and number of physical subsystems.

### Correctable operators

For recovery constructions that start from an error set, use:

```python
Ek = ops.correctable()
```

This returns operators suitable for recovery synthesis (e.g. Leung recovery, below).

## Recovery maps (`Recovery`)

`qudit.noise.Recovery` constructs a recovery channel for a given noise process and code.

### Petz recovery

Petz recovery is built directly from the channel and code:

```python
from qudit.noise import Recovery
from qudit.tools import Fidelity

rec = Recovery.petz(ops, code)
fid = Fidelity.entanglement(rec, ops, code)
print(fid)
```

In the test suite this achieves high entanglement fidelity for the provided parameters.

### Leung recovery

Leung recovery is built from a correctable error set and a code:

```python
Ek = ops.correctable()
rec = Recovery.leung(Ek, code)
fid = Fidelity.entanglement(rec, ops, code)
print(fid)
```

## Fidelity metrics (`Fidelity`)

`qudit.tools.Fidelity` provides helper metrics for evaluating code + recovery performance.

### Entanglement fidelity

Entanglement fidelity is commonly used for QEC benchmarking because it captures average performance on the entire codespace.

```python
from qudit.tools import Fidelity

fid = Fidelity.entanglement(rec, ops, code)
```

Interpretation:

- `fid = 1.0` indicates perfect correction on the codespace (for the modeled noise).
- lower values indicate residual noise after recovery.

## End-to-end example (mirrors `tests/ECC.py`)

```python
import numpy as np
from numpy import linalg as LA

from qudit.noise import Recovery, Process
from qudit.tools import Fidelity

# Define a 2D codespace inside a 16D physical space
code = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0.0],
    ],
    dtype=np.complex64,
)
code /= LA.norm(code, axis=1)[:, None]

# Noise model
ops = Process.GAD(2, 4, Y=0.01, p=0.001)

# Petz recovery
rec_petz = Recovery.petz(ops, code)
fid_petz = Fidelity.entanglement(rec_petz, ops, code)

# Leung recovery
Ek = ops.correctable()
rec_leung = Recovery.leung(Ek, code)
fid_leung = Fidelity.entanglement(rec_leung, ops, code)

print("Petz entanglement fidelity:", fid_petz)
print("Leung entanglement fidelity:", fid_leung)
```

## Practical tips

- Ensure your `code` rows are normalized; many recovery/fidelity computations assume valid statevectors.
- Keep shapes consistent:
  - the codeword length must match the physical Hilbert space dimension of the channel.
- If you change the number of physical subsystems or the local dimension in `Process.*`, update the code embedding dimension accordingly.
