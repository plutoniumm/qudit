# Circuit

`Circuit` provides a high-level way to construct and execute multi-qudit programs without manually tensoring gates. Circuits can be run with both fixed or variable dimensions, and support both statevector and density matrix evolution.

## Creating circuits

A standard circuit is created by specifying the number of wires and the local dimension per wire.

```python
from qudit import Circuit

c = Circuit(wires=2, dim=2)  # 2 qubits
```

`dim=2` here is equivalent to `dim=[2, 2]` for a homogeneous system. The total Hilbert space dimension (width) is `2^2 = 4`. For heterogeneous systems, pass `dim` as a list with one entry per wire.

```python
from qudit import Circuit

# two qubits + two qutrits
c = Circuit(wires=4, dim=[2, 2, 3, 3])
```

Circuit optionally takes in a `mode` argument to specify whether the circuit should be used for statevector (VECTOR) or density matrix (MATRIX) evolution. The default is VECTOR, which is the most common use case.

Circuit also additionally takes in a device which to run on, defaulting to `"cpu"`. If you have a compatible GPU, you can specify `device="cuda"`, or `device="mps"` for Apple Silicon. `qudit` supports all devices supported by [PyTorch](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) with the complex backend.

```python
from qudit import Circuit, Mode

cV = Circuit(wires=2, dim=2, mode=Mode.VECTOR, device="cpu")
cM = Circuit(wires=2, dim=2, mode=Mode.MATRIX, device="mps")
```

## Properties
Circuit contains the following properties and methods:

| Property/Method | Description |
| --- | --- |
| `wires` | Number of wires (particles/dits) |
| `dim` | Local dimension per wire (`list[int] | int`) |
| `device` | Device the circuit is on (e.g. `"cpu"`, `"cuda"`, `"mps"`) |
| `mode` | Whether the circuit is for statevector (VECTOR) or density matrix (MATRIX) evolution |
| `width` | Total Hilbert space dimension (product of local dimensions, $\prod_i \text{d}_i$) |
| `operations` | `list[Frame]` of operations in the circuit |
| `gates` | `dict[int, GateSet]` mapping local dimension to `Gategen` |
| `draw()` | Visualize the circuit diagram |

The circuit primitive is based on the [`nn.Sequential`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html) pattern, so evaluation of the circuit `c` on an input `x` applies the operations in sequence: $c(x) = O_n(...O_2(O_1(x)))$ and is called simply as `c(x)`.

## Gates

Gates are accessed through the circuit’s gate generators by dimension.

```python
G2 = c.gates[2]  # qubit gate set
G3 = c.gates[3]  # qutrit gate set
```

Then use `c.gate(...)` to append operations.

When a circuit is containing unique dimensions `{d_1, d_2, ...}`, the gate sets are accessed as `Gd1 = c.gates[d_1]`, `Gd2 = c.gates[d_2]`, etc. Dimensions which are not present in the circuit will not have a corresponding gate set. For example, if the circuit has `dim=[2, 3]`, then `c.gates[4]` will raise a `KeyError`.

> [!NOTE]
> It is not possible to have gates with `dim=0` or `dim=1` since these do not correspond to valid quantum systems. Attempting to access `c.gates[0]` or `c.gates[1]` will return `None`.

### Simple gates

```python
from qudit import Circuit
import torch as pt

c = Circuit(wires=1, dim=2)
G2 = c.gates[2]

c.gate(G2.H, [0])
c.gate(G2.RY, [0], angle=1.234)

zero = pt.zeros(c.width, dtype=pt.complex64)
zero[0] = 1.0
psi = c(zero)
```

Most two-qudit gates take a list `[control, target]` (or generally `[wire0, wire1]`).

```python
c = Circuit(wires=2, dim=2)
G2 = c.gates[2]

c.gate(G2.H, [0])
c.gate(G2.CX, [0, 1])
```

While you may pass in an arbitrary unitary also, it will not show up as a clean gate in the circuit diagram.

```python
from qudit import Circuit
import torch as pt

c = Circuit(wires=1, dim=3)
G3 = c.gates[3]

cycle = pt.tensor([
  [0, 1, 0],
  [0, 0, 1],
  [1, 0, 0]
], dtype=pt.complex64)
c.gate(cycle, [0])

print(c.draw())
# |0> [3] ┤─None─┤
```

in which case `qudit` includes a wrapper method to convert the unitary into a `Gate` with a generic name `U` or a user-specified name if the unitary has a `name` attribute or is a callable with a `__name__` attribute.

```python
from qudit import Circuit

c = Circuit(wires=1, dim=3)
G3 = c.gates[3]

cycle = G3.U([
  [0, 1, 0],
  [0, 0, 1],
  [1, 0, 0]
], name="CYC")
c.gate(cycle, [0])

print(c.draw())
# |0> [3] ┤─CYC─┤
```

### Mixed-dimension circuits

In mixed systems, use the gate set matching the target pair’s dimension.
For example, for two qutrit wires in a `[2, 2, 3, 3]` circuit, use `G3` on wires 2 and 3.

```python
c = Circuit(wires=4, dim=[2, 2, 3, 3])
G2, G3 = c.gates[2], c.gates[3]

c.gate(G2.H, [0])
c.gate(G2.X, [1])
c.gate(G3.CX, [2, 3])
```

## Running a circuit
Running a circuit is a matter of simple evaluation of the neural net on an input vector. The input should be a valid statevector or density matrix depending on the circuit mode, and should have the correct shape matching the circuit width.

For statevector circuits
```python
import torch
from qudit import Circuit, Mode

def ket0(size):
  x = pt.zeros(size, dtype=pt.complex64)
  x[0] = 1.0
  return x

c = Circuit(wires=1, dim=3, mode=Mode.VECTOR)
G3 = c.gates[3]

c.gate(G3.H, [0])

x = ket0(c.width)

# evaluate on input statevector
psi = c(x)
```

For matrix mode very little changes other than the mode and the input state. The input should be a density matrix, which can be constructed from a statevector `x` as `x @ x.conj().T` (or equivalently `x.view(size, 1) @ x.view(1, size).conj()`).

```python
def to_rho(x):
    size = x.numel()
    return x.view(size, 1) @ pt.conj(x.view(1, size))

c = Circuit(wires=1, dim=3, mode=Mode.VECTOR) # [!code --]
c = Circuit(wires=1, dim=3, mode=Mode.MATRIX) # [!code ++]

# and
rho = c(to_rho(x))
```

> [!TIP]
> For circuits with noise — stochastic Kraus sampling, quantum trajectories, or systematic over-rotation — see [Noisy Circuits](/noisy).

## Parameterized circuits and autograd

Circuit parameters can be wired to `torch.nn.Parameter` values by passing them as gate keyword arguments.
This enables gradient-based optimization when the circuit is used inside a `torch.nn.Module`.

Minimal pattern:

::: code-group

```python [Example]
class Learnable(Hybrid):
    def __init__(self, wires=2, dim=2, device="cpu"):
        super().__init__()
        self.c = Circuit(wires=wires, dim=dim, device=device)
        G = self.c.gates[dim]

        self.theta = nn.Parameter(pt.rand(()))
        self.c.gate(G.RY, [0], angle=self.theta)
        self.c.gate(G.CX, [0, 1])

    def forward(self, x):
        return self.c(x)
```

```python [imports]
from qudit import Circuit
from qudit.ml import Hybrid
import torch.nn as nn
import torch as pt
```

:::

> [!TIP]
> - Prefer creating the circuit once in `__init__` and calling it in `forward`.
> - If you use an encoder (e.g. `nn.Linear`) to map classical data into a valid input vector, ensure the output shape flattens to `c.width`.

With parametrised circuits we encourage using an optimizer to update the parameters, but you can also update them manually by changing the `torch` values.

Additionally be careful when using linear layers between circuits, these are statevectors not measured values.

## Measurement

### Expectation values

`circuit.expectation(operator, state)` runs `state` through the circuit and computes $\langle\psi|O|\psi\rangle$ (VECTOR mode) or $\mathrm{Tr}(O\rho)$ (MATRIX/NOISY mode):

::: code-group

```python [Example]
c = Circuit(wires=1, dim=2)
G = c.gates[2]
c.gate(G.H, [0])   # |+⟩ state

zero = pt.zeros(2, dtype=pt.complex64)
zero[0] = 1.0

Z = pt.tensor([[1, 0], [0, -1]], dtype=pt.complex64)
X = pt.tensor([[0, 1], [1, 0]], dtype=pt.complex64)

print(c.expectation(Z, zero))   # tensor(0.)  ⟨+|Z|+⟩ = 0
print(c.expectation(X, zero))   # tensor(1.)  ⟨+|X|+⟩ = 1
```

```python [imports]
from qudit import Circuit
import torch as pt
```

:::

### Sampling

`circuit.sample(state, shots=1024)` returns a `dict[bitstring, count]` by sampling from the output probability distribution:

::: code-group

```python [Example]
c = Circuit(wires=2, dim=2)
G = c.gates[2]
c.gate(G.H, [0])
c.gate(G.CX, [0, 1])

zero = pt.zeros(4, dtype=pt.complex64)
zero[0] = 1.0

counts = c.sample(zero, shots=1000)
# counts ≈ {"00": 500, "11": 500}
print(counts)
```

```python [imports]
from qudit import Circuit
import torch as pt
```

:::

Bitstrings are mixed-radix for heterogeneous dimensions: a `[2,3]` circuit produces strings like `"02"` (qubit 0, qutrit 2).

## Tips

- Use `c.width` to allocate correctly-sized inputs.
- For mixed dimensions, compute widths as the product of per-wire dimensions.
