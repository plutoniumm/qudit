import sys

sys.path.append("..")

import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam

from qudit.ml import Accel, Hybrid
import qudit.circuit.gates as G
from qudit import Circuit

print(f"Using device: {Accel.available()}")


def train_model(model, target_state, data, epochs=500, lr=0.01):
    optimizer = Adam(model.parameters(), lr=lr)
    target_state = target_state.to(model.device)

    for epoch in range(epochs):
        output = model(data)
        loss = torch.sum(torch.abs(target_state - output))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

class HybridGellMann(Hybrid):
    def __init__(self, dim, wires):
        super().__init__(dim=dim, wires=wires, device="cpu")
        self.circuit = Circuit(wires, dim=dim)

        params = nn.ParameterDict(
            {
                "j0k1": nn.Parameter(torch.rand(wires)),
                "j0k2": nn.Parameter(torch.rand(wires)),
                "j1k2": nn.Parameter(torch.rand(wires)),
            }
        )

        for name, (j, k) in zip(params, [(0, 1), (0, 2), (1, 2)]):
            self.gate(G.GMR, list(range(wires)), j=j, k=k, angle=params[name])

        self.gate(G.CX, [0, 1])
        self.encoder = nn.Linear(200, wires)

D, wires = 5, 2
model = HybridGellMann(dim=D, wires=wires)

vec = torch.zeros(D**wires, dtype=torch.complex64)
vec[[0, D, 2 * D + 2]] = 1.0
target = vec.reshape(-1, 1) / np.sqrt(3)

x0 = np.zeros(D**wires, dtype=np.float32)

train_model(model, target, x0)

print("--------------------------------------")
class HybridQubit(Hybrid):
    def __init__(self, wires):
        super().__init__(dim=2, wires=wires, device="cpu")
        self.circuit = Circuit(wires, dim=self.dim, device=self.device)
        self.full = full = list(range(wires))

        self.angles = nn.ParameterDict(
            {
                "rx": nn.Parameter(torch.rand(wires)),
                "ry": nn.Parameter(torch.rand(wires)),
                "rz": nn.Parameter(torch.rand(wires)),
            }
        )

        self.gate(G.RX, full, angle=self.angles["rx"])
        self.gate(G.RY, full, angle=self.angles["ry"])
        self.gate(G.RZ, full, angle=self.angles["rz"])

        self.gate(G.CX, [0, 1])
        self.encoder = nn.Linear(200, wires)

    def forward(self, x):
        x = self.encoder(x).flatten()

        rz = G.RZ(
            dim=self.dim,
            dits=self.wires,
            index=self.full,
            angle=x,
            device=self.device,
        )

        y = torch.zeros(2**self.wires, dtype=torch.complex64, device=self.device)
        y[0] = 1.0
        x = rz(y)

        return self.circuit(x)


model = HybridQubit(wires=2)

target = torch.tensor([[1], [0], [0], [1]])/ np.sqrt(2)
train_model(model, target, torch.ones((1, 200)))
