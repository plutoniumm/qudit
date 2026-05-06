# Circuit Cutting

Circuit cutting runs a large circuit as two independent sub-circuits by decomposing an entangling gate at a partition boundary. Each sub-circuit runs separately; the results are recombined to recover the full joint probability distribution.

This is useful when a circuit is too wide for a device — cutting trades a single execution of the full circuit for $O(k)$ executions of smaller ones, where $k$ is the number of terms in the gate decomposition.

## Coupler

`Coupler` decomposes a generalized CX gate between a $d_1$-system and a $d_2$-system into a sum of local operators:

$$\text{CX} = \sum_k c_k \, (A_k \otimes B_k)$$

```python
from qudit.cut import Coupler

coupler = Coupler(d1=2, d2=2)              # qubit–qubit
coupler = Coupler(d1=2, d2=3)              # qubit–qutrit
coupler = Coupler(d1=3, d2=3, method="basis")  # Gell-Mann expansion
```

| Parameter | Type | Description |
| --- | --- | --- |
| `d1` | `int` | Local dimension of the control system (≥ 2) |
| `d2` | `int` | Local dimension of the target system (≥ 2) |
| `method` | `str` | `"optimal"` (default): SVD, fewest terms. `"basis"`: Gell-Mann expansion, more terms but interpretable. |
| `threshold` | `float` | Terms with $|c_k| \leq$ threshold are dropped (default `1e-6`). |

### Properties

| Property | Description |
| --- | --- |
| `coupler.CX` | The full $d_1 d_2 \times d_1 d_2$ CX matrix as a `torch.Tensor` |
| `coupler.terms` | `list[(Operator, Operator, float|complex)]` — the decomposition terms |
| `coupler.reconstruct()` | Resum the terms; max error over entries should be $\lesssim 10^{-5}$ |

```python
coupler = Coupler(2, 3)
print(len(coupler.terms))         # number of rank-1 terms

err = (coupler.CX - coupler.reconstruct()).abs().max()
print(err.item())                 # < 1e-5
```

`"optimal"` (SVD) gives real float coefficients. `"basis"` (Gell-Mann) gives complex coefficients because the shift operators $X^r$ are not Hermitian.

## Marking a cut

Call `qc.cut(coupler, [c_A, c_B])` instead of `qc.gate(coupler.CX, [c_A, c_B])`. The gate is not added to the forward pass — it is replaced by the Coupler decomposition when `run_cut()` is called.

```python
from qudit import Circuit
from qudit.cut import Coupler
import torch as pt

coupler = Coupler(2, 2)
qc = Circuit(4, dim=2)
G = qc.gates[2]

qc.gate(G.H, [0])
qc.gate(G.H, [2])
qc.gate(coupler.CX, [0, 1])   # ordinary gate, not cut
qc.cut(coupler, [2, 3])        # marks the cut
qc.gate(G.H, [2])              # post-cut gates are fine

probs = qc.run_cut()           # dict[bitstring, float]
```

`run_cut()` returns a `dict[bitstring, float]` — the same format as `circuit.sample()` but exact (probability, not counts).

Gates on either partition may appear before or after `cut()`. Each sub-circuit is built as: pre-cut gates → cut operator term → post-cut gates.

The partition is determined by the cut position: wires $0 \ldots \text{split}-1$ form partition A, wires $\text{split} \ldots n-1$ form partition B.

## Non-adjacent cuts

For gates acting on wires that are not adjacent ($c_B > c_A + 1$), the partition boundary is ambiguous. Provide `split=` explicitly:

```python
# 5-wire circuit, CX on wires [1, 3]
qc.cut(coupler, [1, 3], split=2)  # partA = {0,1},   partB = {2,3,4}
qc.cut(coupler, [1, 3], split=3)  # partA = {0,1,2}, partB = {3,4}
```

`split` must satisfy `c_A < split <= c_B`. Omitting it for a non-adjacent gate raises `ValueError`. Any gates on wires in the partition that contains the gap wires must stay fully within that partition — a gate spanning both still raises `ValueError`.

## Mixed dimensions

Coupler dimensions must match the wire dimensions at the cut position.

```python
coupler = Coupler(2, 3)                            # d1=2, d2=3
qc = Circuit(4, dim=[2, 3, 2, 3])
G2 = qc.gates[2]

qc.gate(G2.H, [0])
qc.gate(G2.H, [2])
qc.gate(coupler.CX, [0, 1])
qc.cut(coupler, [2, 3])

probs = qc.run_cut()
# bitstrings are mixed-radix: "0012" means wire0=0, wire1=0, wire2=1, wire3=2
```

## Mode support

| Mode | Supported | Notes |
| --- | --- | --- |
| `VECTOR` | yes | Sub-circuits run in VECTOR mode |
| `MATRIX` | yes | Outer circuit is density matrix; sub-circuits reduce to VECTOR (pure-state equivalence) |
| `NOISY` | yes | Sub-circuits run noise-free; cutting produces the noiseless joint distribution |
| `TRAJECTORY` | **no** | Raises `NotImplementedError` — stochastic collapse per term makes amplitude reconstruction invalid |

For `Mode.NOISY`, the cut reconstructs the noise-free distribution. This is an approximation; noisy circuit cutting with per-term noise propagation is not yet supported.

## Limitations

- **Single cut only.** `run_cut()` raises `NotImplementedError` if more than one `cut()` call was made.
- **TRAJECTORY mode.** Raises `NotImplementedError`.
- **Spanning gates.** Any non-cut gate with wires in both partitions raises `ValueError`. Only the marked cut gate may cross the boundary.
- **Cut must target the Coupler's CX.** The Coupler decomposes its own `CX` property. Cutting a custom gate requires building a `Coupler` whose `CX` matches that gate exactly.
- **NOISY post-cut gates.** Post-cut gates run noise-free in sub-circuits regardless of the outer circuit's noise model.

## Full example

::: code-group

```python [Example]
coupler = Coupler(2, 2)
qc_full = Circuit(4, dim=2)
qc_cut  = Circuit(4, dim=2)

for qc in (qc_full, qc_cut):
    G = qc.gates[2]
    qc.gate(G.H, [0])
    qc.gate(G.H, [2])
    qc.gate(coupler.CX, [0, 1])

qc_full.gate(coupler.CX, [2, 3])
qc_cut.cut(coupler, [2, 3])

zero = pt.zeros(16, dtype=pt.complex64)
zero[0] = 1.0

p_full = qc_full(zero).abs().pow(2)
p_cut  = qc_cut.run_cut()

# p_cut ≈ {"0000": 0.25, "0011": 0.25, "1100": 0.25, "1111": 0.25}
```

```python [imports]
from qudit import Circuit
from qudit.cut import Coupler
import torch as pt
```

:::
