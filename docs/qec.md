# Quantum Error Correction (QEC)

`qudit` provides an end-to-end QEC workflow using PyTorch Gate-based channels throughout:

1. Define a code subspace (built-in or from stabilizers)
2. Apply a noise channel to encoded states
3. Construct a recovery map
4. Measure how well recovery works

Main imports:

```python
from qudit.qec import Recovery
from qudit.qec.lib import Dutta3, Leung, Perfect
from qudit.noise import Process
```

---

## Code representation (`Code`)

A `Code` wraps a `torch.Tensor` of shape `(k, d^n)` where each row is one logical codeword as a statevector in the $d^n$-dimensional physical Hilbert space.

| Property/Method | Description |
| --- | --- |
| `code.codewords` | Raw `torch.Tensor` of shape `(k, d^n)` |
| `code.dim` | Number of logical codewords $k$ |
| `code.dits` | Number of physical qudits $n$ |
| `code[i]` | $i$-th codeword as a 1D tensor |
| `code.toTensor()` | Returns all codewords as a tensor (indexable) |
| `Code.isValid(codewords)` | Assert normalization and mutual orthogonality |
| `Code.fromStabilizers(stabs)` | Construct code from Pauli stabilizer strings |

### Built-in codes

Three standard amplitude-damping codes are provided in `qudit.qec.lib`:

| Code | Description |
| --- | --- |
| `Dutta3()` | 3-qubit permutation-invariant code; smallest single-AD-error-correcting code |
| `Leung()` | 4-qubit code; standard single-AD-error correction |
| `Perfect()` | [[5,1,3]] 5-qubit perfect code; corrects any single-qubit error |

```python
from qudit.qec.lib import Leung, Dutta3, Perfect

code = Leung()
state0, state1 = code.toTensor()  # two codeword tensors

print(state0.shape)  # torch.Size([16])  (2^4 = 16)
print(state1.norm()) # 1.0
```

### From stabilizers

`Code.fromStabilizers` constructs the codespace projector from a list of Pauli stabilizer generators and extracts codewords via SVD or randomized range finding:

```python
from qudit.qec.codes import Code

code = Code.fromStabilizers(["ZZZII", "IIZZZ", "XIXXI", "IXXIX"])
print(len(code))  # 2  (= 2^(5-4))
```

---

## Noise channels (`Process`)

`Process` builds multi-qudit noise channels as `Channel` objects with correctable subsets pre-labeled. The channel applies $\Phi(\rho) = \sum_k E_k\rho E_k^\dagger$ where each $E_k$ is a sequence of local `Gate` operators embedded in the full Hilbert space.

```python
from qudit.noise import Process

noise = Process.AD(d=2, n=4, Y=0.1, order=3)
```

| Process | Description |
| --- | --- |
| `Process.AD(d, n, Y, order)` | Amplitude damping; only lowering operators |
| `Process.GAD(d, n, Y, p, order)` | Generalized AD; lowering + raising operators |
| `Process.Pauli(n, paulis, p, order)` | Pauli channel over `{I,X,Y,Z}` words |

Applying a channel to a density matrix:

```python
import torch as pt

def to_rho(psi):
    N = psi.numel()
    return (psi.view(N, 1) @ psi.view(1, N).conj()).to(pt.complex64)

state0, state1 = Leung().toTensor()
rho0 = to_rho(state0)

noisy0 = noise.run(rho0)
```

---

## Recovery maps (`Recovery`)

`Recovery` constructs a recovery `Channel` from a noise channel and a list of codeword tensors. All three constructors have the same signature:

```python
Recovery.petz(channel, codewords)   -> Channel
Recovery.leung(channel, codewords)  -> Channel
Recovery.cafaro(channel, codewords) -> Channel
```

where `codewords` is `List[pt.Tensor]` (each codeword as a 1D tensor).

### Petz recovery

The Petz recovery map minimizes a distinguishability measure and is given by:

$$\mathcal{R}_\mathrm{Petz}: R_k = P\,E_k^\dagger\,[\mathcal{E}(P)]^{-1/2}$$

where $P = \sum_i |\bar{i}\rangle\langle\bar{i}|$ is the code projector.

::: code-group

```python [Example]
state0, state1 = Leung().toTensor()
rho0 = to_rho(state0)

noise = Process.AD(d=2, n=4, Y=0.1, order=3)
noisy0 = noise.run(rho0)

rec = Recovery.petz(noise, [state0, state1])
clean0 = rec.run(noisy0)

fid = pt.real(pt.trace(rho0 @ clean0)).item()
print(f"Fidelity after recovery: {fid:.4f}")  # ~0.9889
```

```python [imports]
from qudit.qec import Recovery
from qudit.qec.lib import Leung
from qudit.noise import Process
import torch as pt

def to_rho(psi):
    N = psi.numel()
    return (psi.view(N, 1) @ psi.view(1, N).conj()).to(pt.complex64)
```

:::

### Leung recovery

The Leung map uses a polar decomposition: for each $k$, factor $E_k P = U_k \Sigma_k V_k^\dagger$ and set $R_k = P U_k^\dagger$:

```python
rec = Recovery.leung(noise, [state0, state1])
```

### Cafaro recovery

The Cafaro map normalizes codeword projections by the overlap $\langle\bar{i}|E_k^\dagger E_k|\bar{i}\rangle$:

```python
rec = Recovery.cafaro(noise, [state0, state1])
```

---

## End-to-end example

::: code-group

```python [Example]
state0, state1 = Leung().toTensor()
rho0, rho1 = to_rho(state0), to_rho(state1)

noise = Process.AD(d=2, n=4, Y=0.1, order=3)

noisy0 = noise.run(rho0)
noisy1 = noise.run(rho1)

rec = Recovery.petz(noise, [state0, state1])

clean0 = rec.run(noisy0)
clean1 = rec.run(noisy1)

fid = lambda r, s: pt.real(pt.trace(r @ s)).item()

print("Noisy  fidelity |0L>:", fid(rho0, noisy0))
print("Recovered fidelity |0L>:", fid(rho0, clean0))
print("Noisy  fidelity |1L>:", fid(rho1, noisy1))
print("Recovered fidelity |1L>:", fid(rho1, clean1))
```

```python [imports]
from qudit.qec import Recovery
from qudit.qec.lib import Leung
from qudit.noise import Process
import torch as pt

def to_rho(psi):
    N = psi.numel()
    return (psi.view(N, 1) @ psi.view(1, N).conj()).to(pt.complex64)
```

:::

> [!NOTE]
> `Recovery` returns a `Channel` object;  apply it via `.run(rho)` exactly as you would any noise channel.

---

## Practical notes

- Codeword tensors must be `torch.Tensor` (1D, on the same device). Call `.toTensor()` on a `Code` and unpack the rows.
- `Process.AD(order=3)` labels correctable Kraus words up to 3-photon-loss order; `Recovery.petz` uses all Kraus words regardless of `correctables`.
- For large codes the Petz pseudo-inverse can be slow; try `Recovery.cafaro` as a faster approximation.
- To validate a custom code before running QEC, use `Code.isValid(code.toTensor())`.
