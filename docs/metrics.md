# Metrics

`qudit.tools` provides classes of information-theoretic functionals: `Fidelity`, `Entropy`, `Info`, and `Distance`.

All inputs can be statevectors (1D arrays) or density matrices (2D arrays). Methods will promote pure states as statevectors, to density matrices automatically.

## Fidelity

`Fidelity` measures overlap between quantum states and channels.

| Method | Description |
| --- | --- |
| `Fidelity.default(rho, sigma)` | Uhlmann fidelity $F(\rho,\sigma)$ |
| `Fidelity.channel(kraus, rho)` | Apply a Kraus channel $\mathcal{E}(\rho)=\sum_k K_k\rho K_k^\dagger$ |
| `Fidelity.entanglement(R, E, codes)` | Entanglement fidelity for encode→noise→recovery |
| `Fidelity.bare_qubit(R, E, state)` | Average fidelity of a single logical state through noise→recovery |
| `Fidelity.cafaro(kraus)` | Cafaro proxy $F_e = \sum_k \|\mathrm{Tr}(K_k)\|^2/N^2$ |
| `Fidelity.negativity(rho, dA, dB)` | Negativity $\mathcal{N}(\rho)=(\|\rho^{T_B}\|_1-1)/2$ |

### State fidelity

For pure statevectors, fidelity is $F = |\langle\psi|\phi\rangle|^2$.
For density matrices, $F(\rho,\sigma)=\big(\mathrm{Tr}\sqrt{\sqrt{\rho}\,\sigma\sqrt{\rho}}\big)^2$.

::: code-group

```python [Example]
psi = Ket("0")
phi = (Ket("0") + Ket("1")).norm()

print(Fidelity.default(psi, phi))  # 0.5
```

```python [imports]
from qudit import Basis
from qudit.tools import Fidelity

Ket = Basis(2)
```

:::

### Entanglement fidelity

Used to benchmark quantum error correction: how well does a code+recovery pipeline preserve the logical subspace under noise?

$$F_e = \langle QR| (\mathcal{R} \circ \mathcal{E})(|QR\rangle\langle QR|) |QR\rangle$$

where $|QR\rangle = \frac{1}{\sqrt{k}}\sum_i |\bar{i}\rangle|i\rangle$ is the purification of the maximally mixed code state.

::: code-group

```python [Example]
ops = Process.GAD(2, 4, Y=0.01, p=0.001)
rec = Recovery.petz(ops, code)

fid = Fidelity.entanglement(rec, ops, code)
print(fid)  # close to 1.0 for small noise
```

```python [imports]
from qudit.noise import Process, Recovery
from qudit.tools import Fidelity
import numpy as np

code = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0.0],
], dtype=np.complex64)
code /= np.linalg.norm(code, axis=1)[:, None]
```

:::

## Entropy

`Entropy` computes various entropic quantities. All density-matrix methods accept statevectors and promote them to $\rho = |\psi\rangle\langle\psi|$.

| Method | Formula | Notes |
| --- | --- | --- |
| `Entropy.neumann(rho)` | $S(\rho) = -\mathrm{Tr}(\rho\log_2\rho)$ | Default entropy |
| `Entropy.shannon(probs)` | $H(p) = -\sum_i p_i \log_2 p_i$ | Classical probabilities |
| `Entropy.tsallis(rho, q)` | $S_q = (1-\sum_i\lambda_i^q)/(q-1)$ | $q\to1$ recovers von Neumann |
| `Entropy.renyi(rho, alpha)` | $S_\alpha = \frac{1}{1-\alpha}\log_2\sum_i\lambda_i^\alpha$ | $\alpha\to1$ recovers von Neumann |
| `Entropy.hartley(probs)` | $H_0 = \log_2\|\mathrm{supp}(p)\|$ | Cardinality of support |
| `Entropy.unified(rho, q, alpha)` | $(q,\alpha)$-entropy family | Interpolates Tsallis/Renyi |
| `Entropy.relative(rho, sigma)` | $D(\rho\|\sigma) = \mathrm{Tr}[\rho(\log\rho-\log\sigma)]$ | Quantum KL divergence |
| `Entropy.conditional(rho, dA, dB)` | $S(A\|B) = S(AB) - S(A)$ | Bipartite state required |

All methods accept an optional `base` keyword (default `2.0`) to change the logarithm base.

::: code-group

```python [Example]
rho = np.array([[0.5, 0], [0, 0.5]])

print(Entropy.neumann(rho))      # 1.0  (maximally mixed qubit)
print(Entropy.tsallis(rho, q=2)) # 0.5  (linear entropy)
print(Entropy.renyi(rho, alpha=2))  # 1.0
```

```python [imports]
from qudit.tools import Entropy
import numpy as np
```

:::

> [!NOTE]
> `Entropy.default` is an alias for `Entropy.neumann`. Shannon entropy takes a probability vector (not a density matrix).

## Info

`Info` derives higher-level correlations from entropies on bipartite states $\rho_{AB}$.

| Method | Formula | Description |
| --- | --- | --- |
| `Info.mutual(rho, dA, dB)` | $I(A:B) = S(A)+S(B)-S(AB)$ | Quantum mutual information |
| `Info.coherent(rho_AB, dA, dB)` | $I_c(A\rangle B) = S(B)-S(AB)$ | Coherent information |
| `Info.conditional(rho, dA, dB)` | see below | Two conventions |

`Info.conditional` has a `true_case` flag:
- `true_case=True` (default): measurement-induced conditional entropy on $B$ given a projective measurement on $A$
- `true_case=False`: algebraic conditional entropy $S(AB)-S(A)$

::: code-group

```python [Example]
# Bell state (maximally entangled)
psi = (np.kron([1, 0], [1, 0]) + np.kron([0, 1], [0, 1])) / np.sqrt(2)
rho = np.outer(psi, psi.conj())

print(Info.mutual(rho, 2, 2))    # 2.0  (maximum for 2 qubits)
print(Info.coherent(rho, 2, 2))  # 1.0
```

```python [imports]
from qudit.tools import Info
import numpy as np
```

:::

## Distance

`Distance` measures distinguishability between quantum states.

| Method | Formula | Description |
| --- | --- | --- |
| `Distance.trace(rho, sigma)` | $\frac{1}{2}\|\rho-\sigma\|_1$ | Operational distinguishability |
| `Distance.bures(rho, sigma)` | $\sqrt{2-2\sqrt{F(\rho,\sigma)}}$ | Metric on density matrices |
| `Distance.jensen_shannon(rho, sigma)` | $\frac{1}{2}(D(\rho\|m)+D(\sigma\|m))$, $m=\frac{\rho+\sigma}{2}$ | Symmetric, bounded in $[0,1]$ |
| `Distance.relative_entropy(rho, sigma)` | $D(\rho\|\sigma) = \mathrm{Tr}[\rho(\log\rho-\log\sigma)]$ | Alias for `Entropy.relative` |

::: code-group

```python [Example]
rho = np.array([[1, 0], [0, 0]], dtype=complex)     # |0><0|
sigma = np.array([[0.5, 0], [0, 0.5]], dtype=complex)  # maximally mixed

print(Distance.trace(rho, sigma))  # 0.5
print(Distance.bures(rho, sigma))  #~0.765
```

```python [imports]
from qudit.tools import Distance
import numpy as np
```

:::

> [!TIP]
> All four classes accept both `np.ndarray` density matrices and 1D pure statevectors;  1D inputs are automatically promoted to rank-1 density matrices where needed.
