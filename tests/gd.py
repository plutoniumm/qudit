import sys

sys.path.append("..")

import torch.nn as nn
import numpy as np
import torch

from qudit.circuit import tensorise
from qudit.ml import Accel, Hybrid
from torch.optim import Adam
from qudit import Circuit

print(f"Using device: {Accel.available()}")

C64 = torch.complex64
dev = "cpu"


def train(model, targ, data, epochs=100, lr=0.01):
    optimizer = Adam(model.parameters(), lr=lr)
    targ = tensorise(targ, device=dev)
    data = tensorise(data, device=dev)

    for epoch in range(epochs):
        loss = torch.norm(targ - model(data))

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")


class HybridGellMann(Hybrid):
    def __init__(self, wires):
        super().__init__()
        self.circuit = C = Circuit(wires, dim=5, device=dev)
        G = C.gates[5]

        self.params = params = nn.ParameterDict(
            {
                "j0k1": nn.Parameter(torch.rand(wires)),
                "j0k2": nn.Parameter(torch.rand(wires)),
                "j1k2": nn.Parameter(torch.rand(wires)),
            }
        )

        for name, (j, k) in zip(params, [(0, 1), (0, 2), (1, 2)]):
            for i in range(wires):
                C.gate(G.GMR, i, j=j, k=k, angle=params[name][i])

        C.gate(G.CX, [0, 1])

    def forward(self, x):
        return self.circuit(x)


D, wires = 5, 2
vec = torch.zeros(D**wires, dtype=C64)
vec[[0, D, 2 * D + 2]] = 1.0
vec /= np.sqrt(3)

train(
    HybridGellMann(wires=wires),
    vec.reshape(-1, 1) / np.sqrt(3),
    np.zeros(D**wires, dtype=np.float32),
)

print("--------------------------------------")


class HybridQubit(Hybrid):
    def __init__(self):
        super().__init__()
        self.wires, self.dim = 2, 2

        self.circuit = C = Circuit(self.wires, dim=self.dim, device=dev)
        G = C.gates[2]
        self.full = range(self.wires)

        self.angles = nn.ParameterDict(
            {
                "rx": nn.Parameter(torch.rand(self.wires)),
                "ry": nn.Parameter(torch.rand(self.wires)),
                "rz": nn.Parameter(torch.rand(self.wires)),
            }
        )

        for i in self.full:
            C.gate(G.RX, i, angle=self.angles["rx"][i])
            C.gate(G.RY, i, angle=self.angles["ry"][i])
            C.gate(G.RZ, i, angle=self.angles["rz"][i])

        C.gate(G.CX, [0, 1])
        self.encoder = nn.Linear(200, C.width)

    def forward(self, x):
        if torch.is_complex(x):
            x = x.abs()

        x = self.encoder(x).flatten()
        x = self.circuit(x)
        return x.reshape(-1, 1)


train(
    HybridQubit(),
    torch.tensor([[1], [0], [0], [1]]) / np.sqrt(2),
    torch.ones((1, 200)),
)

print("--------------------------------------")


class HybridMixed(Hybrid):
    def __init__(self, dim, wires, device="cpu"):
        super().__init__()
        self.circuit = C = Circuit(wires, dim=dim, device=device)
        self.full = range(wires)
        G3 = C.gates[3]
        G5 = C.gates[5]

        self.params = nn.Parameter(torch.rand(2 * wires, device=device) * 2 * np.pi)

        C.gate(G3.GMR, 0, j=1, k=0, angle=self.params[0])
        C.gate(G5.GMR, 1, j=0, k=1, angle=self.params[1])

        # C.gate(G.CX, [0, 1])

    def forward(self, x):
        return self.circuit(x)


x0 = torch.zeros(15, dtype=C64)
x0[0] = 1.0
x0 = x0.reshape(-1, 1)

target = torch.zeros(15, dtype=C64)
target[0] = 1.0
target[14] = 1.0
target /= torch.norm(target)

model = HybridMixed(dim=[3, 5], wires=2, device=dev)

train(model=model, targ=target.reshape(-1, 1), data=x0, lr=0.1)
