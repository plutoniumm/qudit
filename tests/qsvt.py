import sys

sys.path.append("..")
import numpy as np
import torch
from qudit.algo import QSVT
from qudit import Circuit

Cplx = torch.complex64
wires, dim = 2, 2
dev = "cpu"

phis = [0.1, 0.2, 0.3]  # 0.1 x + 0.2 x^2 + 0.3 x^3
A = np.array([[0.1, 0.2], [0.3, 0.4]])

qsvt = QSVT(phis, A, wires, dev)

x0 = torch.zeros((2**wires,), dtype=Cplx, device=dev)
x0[0] = 1.0

circuit = Circuit(wires=wires, dim=dim, device=dev)
for phi in phis:
    circuit.gate(qsvt.PCP(phi), [0, 1])
    circuit.gate(qsvt.U, [0, 1])

circuit = circuit.optimise()
print(circuit(x0))
