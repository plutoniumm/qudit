import sys

sys.path.append("..")

from qudit.circuit import Circuit
import numpy as np
import torch

ms = 1e3

circuit = Circuit(2, dim=[2, 3], device="cpu")

H = circuit.make([[1, 1], [1, -1]] / np.sqrt(2), dim=2)

circuit.gate(H, [0])

state = np.zeros(circuit.width, dtype=np.float32)
state[0] = 1  # |0...0>

state = circuit(state)
predicted = torch.tensor(
    [[1, 0.0, 0.0, 1, 0.0, 0.0]] / np.sqrt(2), dtype=torch.complex64
).T

print(state - predicted)
