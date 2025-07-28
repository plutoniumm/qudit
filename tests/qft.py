import sys

sys.path.append("..")

import numpy as np
import torch
import qudit.circuit.gates as gates
from qudit import Circuit

Cplx = torch.complex64
device = "cpu"


def QFT(wires, dim, device="cpu"):
    N = dim**wires
    omega = np.exp(2j * np.pi / N)
    j = torch.arange(N, device=device).view(-1, 1)
    k = torch.arange(N, device=device).view(1, -1)
    return (torch.pow(omega, j * k) / np.sqrt(N)).to(dtype=Cplx)


def cgate(U, dim=2, device="cpu"):
    P0 = torch.zeros((dim, dim), dtype=Cplx, device=device)
    P0[0, 0] = 1.0
    I = torch.eye(dim, dtype=Cplx, device=device)
    return torch.kron(P0, I) + torch.kron(I - P0, U)


# QFT test
wires, dim = 3, 2
qft3 = QFT(wires, dim)
print(np.array(qft3).round(3))

qft_gate = gates.U(matrix=qft3, dim=dim, wires=wires, device=device)
circuit = Circuit(wires=wires, dim=dim, device=device)
circuit.gate(qft_gate, index=[0, 1, 2])
print("Full Circuit Matrix Shape:", circuit.circuit[0].matrix().shape)

# QPE
n_count, n_targ = 3, 1
total = n_count + n_targ
gg = gates.Gategen(dim=dim, device=device)
ctrl_T = cgate(gg.T, dim, device)
ctrl_S = cgate(gg.S, dim, device)
ctrl_Z = cgate(gg.Z, dim, device)

qpe = Circuit(wires=total, dim=dim, device=device)
qpe.gate(gates.H, index=list(range(n_count)))

for i in range(n_count):
    power = n_count - 1 - i
    ctrl = i
    if power == 0:
        mat = ctrl_T
    elif power == 1:
        mat = ctrl_S
    elif power == 2:
        mat = ctrl_Z
    qpe.gate(gates.U(matrix=mat), index=[ctrl, n_count])

iqft = QFT(n_count, dim).conj().T
qpe.gate(gates.U(matrix=iqft), index=list(range(n_count)))

# Input state |0001>
state = torch.zeros(dim**total, 1, dtype=Cplx, device=device)
state[1] = 1.0

final = qpe(state)
probs = torch.abs(final) ** 2
max_idx = torch.argmax(probs).item()
measured = max_idx >> n_targ
phase_est = measured / (2**n_count)

# Output
print("\n--- QPE Results ---")
print("Unitary: T gate, Eigenvector: |1>, True Phase: 1/8 = 0.125")
print(f"Most probable measurement (decimal): {measured}")
print(f"Estimated Phase: {measured}/8 = {phase_est:.3f}")
