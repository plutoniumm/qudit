# QUBO

This page documents how to formulate and solve Quadratic Unconstrained Binary Optimization (QUBO) problems with `qudit` using `qudit.algo.QAOA`.

The tests in `tests/qubo.py` show two common patterns:

- **Binary QUBO** (Ising/QUBO-style objectives over bits)
- **Discrete spin/clock models** (e.g. 3-state “clock” variables) using a Hamiltonian and a custom objective evaluator

> [!NOTE]
> `QAOA` optimizes circuit parameters using gradient-based optimization (PyTorch autograd) while you provide an *evaluation function* that maps candidate samples to a scalar energy/cost.

---

## QUBO problem format

A QUBO is specified as a dictionary mapping index pairs to coefficients:

- Keys are `(i, j)` tuples (typically with `i <= j`)
- Values are real coefficients

```python
problem = {
    (0, 0): -2,
    (1, 1): -2,
    (2, 2): -2,
    (3, 3): -2,
    (0, 1): 2,
    (1, 2): 2,
    (2, 3): 2,
    (0, 3): 2,
}
```

Interpretation (binary variables `x_i ∈ {0,1}`):

\[
E(\mathbf{x}) = \sum_{i,j} Q_{ij} x_i x_j
\]

In code, a common evaluator for this objective is:

```python
def evaluate(Q: dict, solution: list) -> float:
    energy = 0.0
    sol = {i: bit for i, bit in enumerate(solution)}
    for (i, j), val in Q.items():
        energy += val * sol.get(i, 0) * sol.get(j, 0)
    return energy
```

---

## Solving a binary QUBO with `QAOA`

Minimal pattern:

```python
from qudit.algo import QAOA

qaoa = QAOA(
    d=2,            # local dimension (2 for binary)
    wires=4,        # number of variables
    qubo=problem,   # QUBO dictionary
    layers=5,       # QAOA depth
    device="cpu",
)

result = qaoa.solve(evaluate, steps=50, lr=0.05)
```

### Key arguments

| Argument | Meaning |
| --- | --- |
| `d` | Local dimension per wire (`2` for bits). |
| `wires` | Number of decision variables. |
| `qubo` | QUBO coefficient dict as described above. |
| `layers` | Number of alternating operator layers (QAOA depth). |
| `device` | Torch device string (e.g. `"cpu"`, `"cuda"`, `"mps"`). |

### The `solve(...)` method

`solve` performs parameter optimization and repeatedly queries your evaluator.

Typical parameters:

- `steps`: number of optimizer steps
- `lr`: learning rate
- `hook`: optional callback `hook(loss, step)` for logging

Example hook:

```python
def hook(loss, step):
    print(f"Step {step}: Loss = {loss:.4f}")

result = qaoa.solve(evaluate, steps=50, lr=0.05, hook=hook)
```

---

## Comparing against classical samplers (optional)

The test script compares the QUBO result to a classical simulated annealer (from `neal`).

```python
from neal import SimulatedAnnealingSampler

sampleset = SimulatedAnnealingSampler().sample_qubo(problem)
print("Lowest energy:", sampleset.first.energy)
print("Configuration:", sampleset.first.sample)
```

This is not required to use `qudit`, but it is a convenient sanity-check.

---

## Beyond binary: clock / discrete spin models (d > 2)

`QAOA` can also be used for non-binary discrete variables by increasing the local dimension `d` and providing a Hamiltonian plus a custom evaluator.

In `tests/qubo.py`, a 3-state clock model uses:

- `d=3`
- `wires=6`
- A Hamiltonian list with terms of the form `(coefficient, op_string, wires)`

Example structure:

```python
neighbors = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
J, g = 1.0, 0.5

ham = []
for i, j in neighbors:
    ham.append((-J, "ZZ", [i, j]))
for i in range(6):
    ham.append((-g, "Z", [i]))
    ham.append((-g, "Z", [i]))

qaoa = QAOA(
    d=3,
    wires=6,
    hamiltonian=ham,
    offset=0.0,
    layers=3,
    device="cpu",
)
```

Then define an energy function for discrete states `s_i ∈ {0,1,2}`. The test uses a cosine clock-energy:

```python
import torch as pt

def eval_clock(_unused, solution: list) -> float:
    E = 0.0
    for i, j in neighbors:
        D = (solution[i] - solution[j]) % 3
        E += -J * pt.cos(2 * pt.pi * D / 3)
    for s in solution:
        E += -g * 2 * pt.cos(2 * pt.pi * s / 3)
    return float(E)

res = qaoa.solve(eval_clock, steps=10, lr=0.1)
```

> [!TIP]
> For non-binary models, ensure your evaluator matches the variable domain implied by `d`.

---

## `ClockSolver`: matrix-exponentiation approach

`ClockSolver` is an alternative to `QAOA` designed for discrete-variable problems with $d \geq 2$. Instead of decomposing the time-evolution into individual gates, it directly exponentiates the Hamiltonian matrices:

$$U_P(\gamma) = e^{-i\gamma H_P}, \qquad U_B(\beta) = e^{-i\beta H_B}$$

where $H_B = -\sum_i (X_d^{(i)} + X_d^{(i)\dagger})$ is the clock-model mixer. This avoids the gate decomposition overhead and is typically more accurate for higher-dimensional systems.

::: code-group

```python [Example]
from qudit.algo import ClockSolver

neighbors = [(0, 1), (1, 2), (2, 0)]
J = 1.0

ham = [(-J, "ZZ", list(e)) for e in neighbors]

solver = ClockSolver(
    d=3,
    wires=3,
    hamiltonian=ham,
    offset=0.0,
    layers=2,
    device="cpu",
)

result = solver.solve(steps=200, lr=0.1)
print(result["solution"])        # list of ints in {0,1,2}
print(result["probabilities"])   # probability vector of length d^wires
```

```python [imports]
from qudit.algo import ClockSolver
```

:::

### `QAOA` vs `ClockSolver`

| | `QAOA` | `ClockSolver` |
| --- | --- | --- |
| Mixer | RX gate per wire | $e^{-i\beta H_B}$ (matrix exp) |
| Phase sep. | RZ / RZZ gate decomp | $e^{-i\gamma H_P}$ (matrix exp) |
| Works for | $d = 2$, limited $d > 2$ | any $d \geq 2$ |
| Output | binary / d-ary string | d-ary string |
| `solve` returns | `solution`, `probabilities`, `value` | `solution`, `probabilities`, `value` |

> [!TIP]
> For clock/Potts models ($d \geq 3$) `ClockSolver` is the recommended solver. For standard binary QUBO problems ($d=2$), either works; `QAOA` is slightly faster per step due to gate-level sparsity.

---

## Low-level API: `QUBO.toHamiltonian`

`QUBO.toHamiltonian(Q)` converts a QUBO dict into an Ising Hamiltonian list for use with `QAOA` or `ClockSolver`. It maps binary variables $x_i \in \{0,1\}$ to spin variables $s_i = 1-2x_i \in \{+1,-1\}$ via the standard substitution:

$$E(\mathbf{x}) = \sum_{i,j} Q_{ij}\,x_i x_j \;\longrightarrow\; H = \sum_i h_i Z_i + \sum_{i<j} J_{ij} Z_i Z_j + \text{offset}$$

Returns `(ham, offset)` where `ham` is a list of `(coefficient, op_string, wires)` tuples and `offset` is a constant energy shift.

```python
from qudit.algo import QUBO

Q = {
    (0, 0): -2.0,
    (1, 1): -2.0,
    (0, 1):  4.0,
}

ham, offset = QUBO.toHamiltonian(Q)
# ham   = [(coeff, "Z", [i]) for diagonal terms, (coeff, "ZZ", [i,j]) for off-diagonal]
# offset = constant shift
```

Off-diagonal entries are symmetrized: `Q[(i,j)]` and `Q[(j,i)]` are combined. Passing `{}` returns `([], 0.0)`.

---

## Troubleshooting

- **Dimension mismatch**: For binary QUBO problems, use `d=2` and `wires == number of variables`.
- **Slow optimization / unstable loss**: reduce `lr`, increase `steps`, or increase `layers` gradually.
- **Device issues**: if complex ops are unsupported on your accelerator, fall back to `device="cpu"`.
