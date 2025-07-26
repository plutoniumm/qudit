import sys

sys.path.append("..")

import qudit.circuit.gates as G
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
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")


class HybridGellMann(Hybrid):
    def __init__(self, wires):
        super().__init__()
        self.circuit = C = Circuit(wires, dim=5, device=dev)

        params = nn.ParameterDict(
            {
                "j0k1": nn.Parameter(torch.rand(wires)),
                "j0k2": nn.Parameter(torch.rand(wires)),
                "j1k2": nn.Parameter(torch.rand(wires)),
            }
        )

        for name, (j, k) in zip(params, [(0, 1), (0, 2), (1, 2)]):
            C.gate(G.GMR, range(wires), j=j, k=k, angle=params[name])

        C.gate(G.CX, [0, 1])

    def forward(self, x):
        return self.circuit(x)


D, wires = 5, 2
vec = torch.zeros(D**wires, dtype=C64)
vec[[0, D, 2 * D + 2]] = 1.0

train(
    HybridGellMann(wires=wires),
    vec.reshape(-1, 1) / np.sqrt(3),
    np.zeros(D**wires, dtype=np.float32),
)

print("--------------------------------------")


class HybridQubit(Hybrid):
    def __init__(self):
        super().__init__()
        self.circuit = C = Circuit(2, dim=2, device=dev)
        self.full = full = range(wires)
        self.wires, self.dim = wires, 2

        self.angles = nn.ParameterDict(
            {
                "rx": nn.Parameter(torch.rand(wires)),
                "ry": nn.Parameter(torch.rand(wires)),
                "rz": nn.Parameter(torch.rand(wires)),
            }
        )

        C.gate(G.RX, full, angle=self.angles["rx"])
        C.gate(G.RY, full, angle=self.angles["ry"])
        C.gate(G.RZ, full, angle=self.angles["rz"])

        C.gate(G.CX, [0, 1])
        self.encoder = nn.Linear(200, wires)

    def forward(self, x):
        x = self.encoder(x).flatten()

        rz = G.RZ(
            dim=self.dim,
            wires=self.wires,
            index=self.full,
            angle=x,
            device=dev,
        )

        y = torch.zeros(2**self.wires, dtype=C64, device=dev)
        y[0] = 1.0
        x = rz(y)

        return self.circuit(x)


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

        self.params = nn.Parameter(torch.rand(2 * wires, device=device) * 2 * np.pi)

        C.gate(G.GMR, self.full, j=1, k=0, angle=self.params[:wires])
        C.gate(G.GMR, self.full, j=0, k=1, angle=self.params[wires:])
        C.gate(G.CX, [0, 1])

    def forward(self, x):
        return self.circuit(x)


D_list = [3, 5]
dtot = np.prod(D_list)
x0 = torch.zeros(dtot, dtype=C64)
x0[0] = 1.0
x0 = x0.reshape(-1, 1)

target = torch.zeros(dtot, dtype=C64)
target[0] = 1.0
target[14] = 1.0
target /= torch.norm(target)

model = HybridMixed(dim=D_list, wires=2, device=dev)

train(model=model, targ=target.reshape(-1, 1), data=x0, epochs=200, lr=0.1)
