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
| `d` | Local dimension per site |
| `n` | Number of physical sites |
| `Y` | Damping parameter $Y \in [0,1]$ |
| `order` | Max correctable error order (default `1`) |
| `group` | If `True`, keep correctable sets grouped by order |
| `iid` | If `True`, return `Multiplex` of single-wire channels (qubit only) |

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

## Correctable subsets

After building a channel, `channel.correctables` holds a flat list (or grouped list if `group=True`) of Kraus-word indices that are considered correctable up to the given order.

```python
channel = Process.GAD(2, 4, Y=0.01, p=0.001)
Ek = channel.correctable()  # list of Kraus operator lists
```

These are passed directly to `Recovery.leung` for constructing recovery maps (see [Error Correction](/qec)).

## IID noise

`IID` builds single-wire channels and multiplexes them for independent noise per qubit:

| Factory | Description |
| --- | --- |
| `IID.AD(n, y)` | Amplitude damping on each of `n` qubits independently |
| `IID.GAD(n, y, p)` | Generalized amplitude damping on each qubit independently |

> [!TIP]
> Use `Process.GAD(d=2, n=..., iid=True)` as a shortcut;  it delegates to `IID.GAD` and returns a `Multiplex`.
