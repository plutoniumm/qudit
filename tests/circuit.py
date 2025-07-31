import sys

sys.path.append("..")

from unittest import TestCase, main
import qudit.circuit.gates as gates
from qudit import Circuit
import numpy as np
import torch

dev = "cpu"
C64 = torch.complex64


def ket0(size):
    x = torch.zeros(size, dtype=C64, device=dev)
    x[0] = 1.0
    return x


# --- 1. Two-Qubit Bell State Circuit ---
c1 = Circuit(wires=2, dim=2, device=dev)
G2 = c1.gates[2]
c1.gate(G2.H, [0])
c1.gate(G2.CX, [0, 1])

x1 = ket0(c1.width)
output1 = c1(x1)

print(np.round(output1.cpu().numpy().flatten(), 3))

# --- 2. Mixed-Dimension Entanglement Circuit ---
c2 = Circuit(wires=4, dim=[2, 2, 3, 3], device=dev)
G2 = c2.gates[2]
G3 = c2.gates[3]

c2.gate(G2.H, [0])
c2.gate(G2.X, [1])
c2.gate(G3.CX, [2, 3])

x2 = ket0(c2.width)
output2 = c2(x2)

print("Output State (non-zero elements shown):")
non_zero_indices = torch.where(abs(output2) > 1e-6)[0]
for idx in non_zero_indices:
    print(f"  Index {idx.item()}: {output2[idx].item():.3f}")

# --- 3. Three-Qutrit GHZ State Circuit ---
c3 = Circuit(wires=3, dim=3, device=dev)
G3 = c3.gates[3]
c3.gate(G3.H, [0])
c3.gate(G3.CX, [0, 1])
c3.gate(G3.CX, [0, 2])

x3 = ket0(c3.width)
output3 = c3(x3)

print("Output State (GHZ State, non-zero elements shown):")
non_zero_indices_3 = torch.where(abs(output3) > 1e-6)[0]
for idx in non_zero_indices_3:
    print(f"  Index {idx.item()}: {output3[idx].item():.3f}")
