# Noise Channels

`qudit.noise` provides building blocks for quantum noise models: `Channel` holds a Kraus decomposition, `Multiplex` sequences multiple channels, and the `Process` factory constructs standard multi-qudit noise models.

## Channel

A `Channel` represents a completely positive map in Kraus form:

$$\Phi(\cdot) = \sum_k E_k\,(\cdot)\,E_k^\dagger$$

Each Kraus operator $E_k$ is stored as a list of local `Gate` objects;  an operator "word" that is applied left-to-right via `Gate.forwardd`.

| Property/Method | Description |
| --- | --- |
| `ops` | List of Kraus words (`list[list[Gate]]`) |
| `d` | Full Hilbert space dimension |
| `run(rho)` | Apply $\Phi(\rho) = \sum_k E_k\rho E_k^\dagger$ |
| `correctable()` | Return Kraus words listed in `correctables` |
| `isTP` | $\sum_k E_k^\dagger E_k = I$? |
| `isCP` | Choi matrix $\succeq 0$? |
| `isCPTP` | both CP and TP? |
| `toChoi()` | Choi–Jamiołkowski matrix $J(\Phi)$ |
| `toSuperop()` | Superoperator $S$ s.t. $\mathrm{vec}(\Phi(\rho)) = S\,\mathrm{vec}(\rho)$ |
| `toStinespring()` | Stinespring isometry $V$ s.t. $\Phi(\rho)=\mathrm{Tr}_\mathrm{env}(V\rho V^\dagger)$ |

### Running a channel

::: code-group

```python [Example]
# Amplitude damping on 2 qubits
channel = Process.AD(d=2, n=2, Y=0.1)

# Apply to |00><00|
rho = np.zeros((4, 4), dtype=complex)
rho[0, 0] = 1.0

rho_out = channel.run(rho)
print(rho_out.shape)   # (4, 4)
```

```python [imports]
from qudit.noise import Process
import numpy as np
```

:::

### Channel analysis

::: code-group

```python [Example]
channel = Process.AD(d=2, n=2, Y=0.1)

print(channel.isTP)    # True
print(channel.isCP)    # True
print(channel.isCPTP)  # True

J = channel.toChoi()   # shape (d^2, d^2)
S = channel.toSuperop()
V = channel.toStinespring()  # isometry: V†V ≈ I
```

```python [imports]
from qudit.noise import Process
```

:::

> [!NOTE]
> `isTP`, `isCP`, `isCPTP` are `cached_property` values;  they are computed once on first access and cached.


## Multiplex

`Multiplex` sequences multiple channels, applying each in turn:

$$\rho \mapsto \Phi_n \circ \cdots \circ \Phi_2 \circ \Phi_1(\rho)$$

The main use case is independent identically distributed (IID) noise, where each wire gets its own channel applied sequentially.

::: code-group

```python [Example]
# IID amplitude damping on 3 qubits
multi = IID.AD(n=3, y=0.05)

rho = np.zeros((8, 8), dtype=complex)
rho[0, 0] = 1.0

rho_out = multi.run(rho)
```

```python [imports]
from qudit.noise import IID
import numpy as np
```

:::

`Multiplex` flattens nested `Multiplex` inputs on construction, so `Multiplex([multi1, multi2])` gives a single flat list of channels.

## Process

`Process` constructs standard multi-qudit noise channels with correctable subsets pre-labeled.

### Amplitude damping (`AD`)

Pure amplitude damping uses only lowering operators $A_k$ ($p=0$ case of GAD):

$$A_k: |r\rangle \mapsto \binom{r}{k}^{1/2}(1-Y)^{(r-k)/2} Y^{k/2} |r-k\rangle$$

```python
channel = Process.AD(d=2, n=4, Y=0.01, order=1)
```

| Argument | Description |
| --- | --- |
| `d` | Local dimension per site (any `d ≥ 2`) |
| `n` | Number of physical sites |
| `Y` | Damping parameter $Y \in [0,1]$ |
| `order` | Max correctable error order (default `1`) |
| `group` | If `True`, keep correctable sets grouped by order |
| `iid` | If `True`, return `Multiplex` of single-wire channels |

### Generalized amplitude damping (`GAD`)

GAD adds raising operators $R_k$ parameterized by environment excitation probability $p$:

```python
channel = Process.GAD(d=2, n=4, Y=0.01, p=0.001, order=1)
```

Same arguments as `AD`, plus `p` (environment excitation probability).

### Pauli channel

::: code-group

```python [Example]
px, py, pz = 0.01, 0.005, 0.01
channel = Process.Pauli(n=3, paulis=["X", "Y", "Z"], p=[px, py, pz], order=1)
```

```python [imports]
from qudit.noise import Process
```

:::

Kraus operators are tensor products of $\{I, \sqrt{p_X}X, \sqrt{p_Y}Y, \sqrt{p_Z}Z\}$ across all $n$ sites. Correctable subsets are labeled by Hamming weight.

### Depolarising channel

Applies all $d^2$ Weyl-Heisenberg operators as Kraus terms. Works for any $d \geq 2$:

$$\Phi(\rho) = (1-p)\rho + \frac{p}{d^2}\sum_{j,k} W_{jk}\rho W_{jk}^\dagger$$

```python
channel = Process.Depolarising(d=2, n=2, p=0.05)
channel = Process.Depolarising(d=3, n=1, p=0.02)  # qutrit
```

### Phase damping

Kills off-diagonal coherences without energy exchange. For $d=2$: $K_0 = \mathrm{diag}(1,\sqrt{1-p})$, $K_1 = \mathrm{diag}(0,\sqrt{p})$. Works for any $d$:

```python
channel = Process.PhaseDamp(d=2, n=2, p=0.1)
```

### Bit-flip and phase-flip channels

Qudit generalizations using the cyclic shift $X_d$ and clock $Z_d$ operators:

```python
channel = Process.BitFlip(d=2, n=2, p=0.05)   # applies X_d with prob p
channel = Process.PhaseFlip(d=2, n=2, p=0.05)  # applies Z_d with prob p
```

Both reduce to the standard qubit channels when `d=2`.

### Reset channel

Collapses each qudit to $|0\rangle$ with probability $p$:

$$\Phi(\rho) = (1-p)\rho + p\,|0\rangle\langle 0|\,\mathrm{Tr}(\rho)$$

```python
channel = Process.Reset(d=2, n=2, p=0.1)
```

### Thermal relaxation

Combined $T_1$ energy decay and $T_2$ dephasing for qubits ($d=2$ only). Requires $T_2 \leq 2T_1$:

```python
channel = Process.ThermalRelax(n=2, T1=100e-6, T2=80e-6, t=50e-9)
```

All `Process` channels accept an `iid=True` flag to return a `Multiplex` of independent single-site channels instead of a joint $n$-site channel.

### Weyl-Heisenberg channel (NoisyGate)

For circuit-level noise in `Mode.NOISY`, `NoisyGate("weyl", param, ...)` implements the Heisenberg-Weyl displacement channel for any local dimension $d$:

$$\Phi(\rho) = (1 - \textstyle\sum_{(m,n)\neq(0,0)} p_{mn})\,\rho + \sum_{(m,n)\neq(0,0)} p_{mn}\, W_{mn}\,\rho\,W_{mn}^\dagger$$

where $W_{mn} = X_d^m Z_d^n$, $X_d$ is the cyclic shift, and $Z_d$ is the clock operator. `param` is a length-$(d^2-1)$ tensor of probabilities in row-major order $(0,1),(0,2),\ldots,(d-1,d-1)$ excluding $(0,0)$.

```python
from qudit.circuit.gates import NoisyGate
import torch

# Qubit Weyl (d=2): 3 parameters for W_01=Z, W_10=X, W_11=XZ
ng2 = NoisyGate("weyl", torch.tensor([0.02, 0.01, 0.01]), index=0, wires=1, dims=2)

# Qutrit Weyl (d=3): 8 parameters for all (m,n) ≠ (0,0)
ng3 = NoisyGate("weyl", torch.tensor([0.005]*8), index=0, wires=1, dims=3)
```

## Correctable subsets

After building a channel, `channel.correctables` holds a flat list (or grouped list if `group=True`) of Kraus-word indices that are considered correctable up to the given order.

```python
channel = Process.GAD(2, 4, Y=0.01, p=0.001)
Ek = channel.correctable()  # list of Kraus operator lists
```

These are passed directly to `Recovery.leung` for constructing recovery maps (see [Error Correction](/qec)).

## IID noise

`IID` builds single-wire channels and multiplexes them for independent identically distributed noise. Both factories work for any local dimension `d ≥ 2`:

| Factory | Signature | Description |
| --- | --- | --- |
| `IID.AD` | `(n, d, y)` | Amplitude damping on each of `n` qudits independently |
| `IID.GAD` | `(n, d, y, p)` | Generalized amplitude damping on each qudit independently |
| `IID.Depolarising` | `(n, d, p)` | Depolarising channel on each qudit independently |
| `IID.PhaseDamp` | `(n, d, p)` | Phase damping on each qudit independently |
| `IID.BitFlip` | `(n, d, p)` | Bit-flip ($X_d$) on each qudit independently |
| `IID.PhaseFlip` | `(n, d, p)` | Phase-flip ($Z_d$) on each qudit independently |
| `IID.Reset` | `(n, d, p)` | Reset to $|0\rangle$ on each qudit independently |

::: code-group

```python [Example]
qubit = IID.AD(n=3, d=2, y=0.05)

qutrit = IID.AD(n=2, d=3, y=0.1)
```

```python [imports]
from qudit.noise import IID
```

:::

> [!TIP]
> Use `Process.AD(d=..., n=..., iid=True)` as a shortcut; it delegates to `IID.AD` and returns a `Multiplex`.
