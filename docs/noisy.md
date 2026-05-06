# Noisy Circuits

`qudit` supports stochastic noise simulation via `Mode.NOISY` and `Mode.TRAJECTORY`, each backed by a pluggable noise model object from `qudit.noise`. Noise is injected per-gate at circuit-construction time; the forward pass samples from the resulting Kraus stacks.

## Mode comparison

| Mode | Input | Output | Noise | Memory |
| --- | --- | --- | --- | --- |
| `VECTOR` | statevector $\|\psi\rangle$ | statevector | none | $O(d^n)$ |
| `MATRIX` | density matrix $\rho$ | density matrix | none (use `Channel` directly) | $O(d^{2n})$ |
| `NOISY` | density matrix $\rho$ | density matrix | stochastic Kraus sample per gate | $O(d^{2n})$ |
| `TRAJECTORY` | statevector $\|\psi\rangle$ | statevector | stochastic quantum jump per gate | $O(d^n)$ |

`NOISY` and `TRAJECTORY` are both **stochastic single-shot** simulations. Average many shots to recover the full channel. `TRAJECTORY` is memory-efficient for large systems because it keeps a statevector instead of a density matrix; averaging $|\psi\rangle\langle\psi|$ over shots approximates $\rho$.

For noiseless density-matrix simulation with an explicit Kraus channel applied once (not per gate), use `Mode.MATRIX` together with `Channel.run(rho)` from [`qudit.noise`](/noise).

## Noise model objects

Import all three from `qudit.noise`:

```python
from qudit.noise import PhysicalNoise, WeylNoise, CoherentNoise
```

### `WeylNoise(p, seed=None)`

Weyl-Heisenberg depolarizing channel. Works for any local dimension $d$. `p` is the total error probability spread uniformly over the $d^2 - 1$ non-identity Weyl operators $W_{mn} = X_d^m Z_d^n$.

| Parameter | Type | Description |
| --- | --- | --- |
| `p` | `float` | Total error probability (0 ≤ p ≤ 1) |
| `seed` | `int \| None` | RNG seed for reproducibility |

::: code-group

```python [Example]
c = Circuit(wires=2, dim=2, mode=Mode.NOISY, noise=WeylNoise(p=0.05))
G = c.gates[2]

c.gate(G.H, [0])
c.gate(G.CX, [0, 1])

zero = pt.zeros(4, dtype=pt.complex64)
zero[0] = 1.0
rho_in = to_rho(zero)

rho_out = c(rho_in)  # single stochastic shot
print(pt.trace(rho_out).real)  # 1.0
```

```python [imports]
from qudit import Circuit, Mode
from qudit.noise import WeylNoise
import torch as pt

def to_rho(x):
    size = x.numel()
    return x.view(size, 1) @ pt.conj(x.view(1, size))
```

:::

### `PhysicalNoise(T1, T2, method='free', gate_times=None, seed=None)`

$T_1$/$T_2$-based Lindblad channel derived from physical relaxation and dephasing times. Kraus operators are cached by `(gate_name, wire, d)` and reused across shots.

| Parameter | Type | Description |
| --- | --- | --- |
| `T1` | `float \| list[float]` | Energy relaxation time(s) in seconds. Scalar or one per wire. |
| `T2` | `float \| list[float]` | Dephasing time(s) in seconds. Scalar or one per wire. Must satisfy $T_2 \leq 2T_1$. |
| `method` | `str` | `'free'` (default): free decoherence, no drive Hamiltonian. `'lindblad'`: full Lindblad with drive (reserved, not yet implemented). |
| `gate_times` | `dict \| None` | Override default gate durations. Defaults: 1-qudit gates 50 ns, 2-qudit gates 200 ns. |
| `seed` | `int \| None` | RNG seed for reproducibility |

::: code-group

```python [Example]
# Uniform T1/T2 across all wires
c = Circuit(wires=2, dim=2, mode=Mode.TRAJECTORY,
            noise=PhysicalNoise(T1=50e-6, T2=30e-6))
G = c.gates[2]

c.gate(G.H, [0])
c.gate(G.CX, [0, 1])

zero = pt.zeros(4, dtype=pt.complex64)
zero[0] = 1.0

psi_out = c(zero)  # single stochastic ket
```

```python [imports]
from qudit import Circuit, Mode
from qudit.noise import PhysicalNoise
import torch as pt
```

:::

Per-wire $T_1$/$T_2$ values let you model heterogeneous hardware:

```python
noise = PhysicalNoise(
    T1=[50e-6, 40e-6],
    T2=[30e-6, 25e-6],
)
c = Circuit(wires=2, dim=2, mode=Mode.NOISY, noise=noise)
```

### `CoherentNoise(eps)`

Systematic over-rotation. Each gate $U$ is perturbed as $U \to U \cdot e^{-i\,\varepsilon\,G}$ where $G$ is a random GUE matrix drawn once per `c.gate(...)` call. Deterministic — no seed needed.

| Parameter | Type | Description |
| --- | --- | --- |
| `eps` | `float` | Perturbation magnitude. Small values ($\lesssim 0.05$) produce near-unitary errors. |

```python
from qudit.noise import CoherentNoise

c = Circuit(wires=1, dim=2, mode=Mode.NOISY, noise=CoherentNoise(eps=0.01))
c.gate(c.gates[2].H, [0])
```

Unlike `WeylNoise` and `PhysicalNoise`, coherent errors preserve the purity of the output state — the circuit remains unitary up to a small systematic rotation rather than introducing classical uncertainty.

## Averaging shots to recover the full channel

Single shots are stochastic. To estimate the full mixed-state output, average density matrices (NOISY) or outer products of kets (TRAJECTORY) over many runs:

::: code-group

```python [NOISY averaging]
shots = 500
rho_avg = pt.zeros((4, 4), dtype=pt.complex64)

for _ in range(shots):
    rho_avg += c(rho_in)

rho_avg /= shots
print(f"purity = {pt.trace(rho_avg @ rho_avg).real:.4f}")
```

```python [TRAJECTORY averaging]
shots = 500
rho_avg = pt.zeros((4, 4), dtype=pt.complex64)

for _ in range(shots):
    psi = c(zero)
    rho_avg += pt.outer(psi, psi.conj())

rho_avg /= shots
```

```python [imports]
from qudit import Circuit, Mode
from qudit.noise import WeylNoise, PhysicalNoise
import torch as pt

def to_rho(x):
    size = x.numel()
    return x.view(size, 1) @ pt.conj(x.view(1, size))

zero = pt.zeros(4, dtype=pt.complex64)
zero[0] = 1.0
rho_in = to_rho(zero)
```

:::

More shots reduce statistical noise; convergence rate is $O(1/\sqrt{\text{shots}})$.

## Seeding for reproducibility

Pass `seed` to `WeylNoise` or `PhysicalNoise` to get identical shot sequences across runs:

```python
noise = WeylNoise(p=0.1, seed=42)
c = Circuit(wires=1, dim=2, mode=Mode.NOISY, noise=noise)
c.gate(c.gates[2].H, [0])

rho_a = c(rho_in)
# re-create with the same seed → identical result
noise2 = WeylNoise(p=0.1, seed=42)
c2 = Circuit(wires=1, dim=2, mode=Mode.NOISY, noise=noise2)
c2.gate(c2.gates[2].H, [0])

rho_b = c2(rho_in)
print(pt.allclose(rho_a, rho_b))  # True
```

`CoherentNoise` is always deterministic (the GUE draw is fixed at gate-add time).

## Qudit-general noise

All three noise models work for any local dimension $d$. Set `dim` on the circuit:

::: code-group

```python [Qutrit WeylNoise]
c = Circuit(wires=2, dim=3, mode=Mode.NOISY, noise=WeylNoise(p=0.05))
G = c.gates[3]

c.gate(G.H, [0])
c.gate(G.CX, [0, 1])

zero = pt.zeros(9, dtype=pt.complex64)
zero[0] = 1.0
rho_out = c(to_rho(zero))
```

```python [Qutrit PhysicalNoise]
c = Circuit(wires=2, dim=3, mode=Mode.TRAJECTORY,
            noise=PhysicalNoise(T1=50e-6, T2=30e-6))
G = c.gates[3]

c.gate(G.H, [0])

zero = pt.zeros(9, dtype=pt.complex64)
zero[0] = 1.0
psi_out = c(zero)
```

```python [imports]
from qudit import Circuit, Mode
from qudit.noise import WeylNoise, PhysicalNoise
import torch as pt

def to_rho(x):
    size = x.numel()
    return x.view(size, 1) @ pt.conj(x.view(1, size))
```

:::

The Weyl-Heisenberg operators scale naturally to dimension $d$: for $d=3$ there are $3^2 - 1 = 8$ non-identity basis operators. `WeylNoise` handles this automatically given `p`.

## Noise model summary

| Model | Stochastic | Works for | Captures | Seed |
| --- | --- | --- | --- | --- |
| `WeylNoise` | yes | any $d$ | incoherent depolarizing | yes |
| `PhysicalNoise` | yes | any $d$ | $T_1$/$T_2$ relaxation/dephasing | yes |
| `CoherentNoise` | no (deterministic) | any $d$ | systematic over-rotation | n/a |

> [!NOTE]
> For Kraus channels applied to a state once (not per gate), use `Channel` and `Process` from [`qudit.noise`](/noise). These are separate from the per-gate noise models described here.
