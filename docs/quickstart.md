# Qudit Quickstart

Get up and running with `qudit`, a Python library for working with *d*-level quantum systems (qudits): circuits, error correction, and QML style workflows.

## Installation
I recommend using a virtual environment (e.g., `venv` or `conda`) to manage your Python dependencies since `qudit` will install `torch` which may conflict with any existing global installations. Install from PyPI:

```bash
pip install qudit
```

If you’re working in a notebook:

```bash
python -m pip install qudit
```

## Verify your install

```bash
python -c "from qudit.random import random_state; print(random_state(2))"
```

## Core ideas

- A **qudit** has dimension **d** (e.g., d=2 qubit, d=3 qutrit).
- A **state** can be represented as a complex vector $|\psi\rangle$ or density matrix $\rho$.
- An **operator** is a $d\times d$ matrix (unitary gates, observables, channels).
- A **register** of *n* qudits has dimension $d^n$.

## Circuit-less Bell state

```python
from qudit import Basis, State
from qudit.circuit import Gategen

G = Gategen(2)
Ket = Basis(2)

Psi = Ket("00")
print(Psi)

HI = G.H ^ G.I
CX = G.CX

# CX H⊗I |00> = (|00> + |11>) / sqrt(2)
Psi = CX @ HI @ Psi
# [0.7071, 0, 0, 0.7071]
print(Psi)
```

## Circuit Bell state

```python
from qudit import Basis, State
from qudit.circuit import Circuit


Ket = Basis(2)
C = Circuit(wires=2, dim=2)
G = C.gates[2]
# or G = Gategen(2)

Psi = Ket("00")

C.gate(G.H, [0])
C.gate(G.CX, [0, 1])
Psi = C(Psi)
```

Should you choose to print the circuit with `print(C.draw())`, you will see a nice ASCII representation of the circuit.

```
|0> [2] ┤─H─╭●─┤
|0> [2] ┤───╰U─┤
```

Which shows us we start with two dits of dimension two, apply a Hadamard gate to the first qudit, and then a CNOT gate between the first and second qudit.