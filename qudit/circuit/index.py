from dataclasses import dataclass
from typing import Union, List
from . import gates as GG

import torch.nn as nn
import numpy as np
import torch

C64 = torch.complex64

@dataclass
class Gateless:
    index: Union[None, List[int]]
    dim: Union[int, List[int]]
    wires: int

    def __init__(
        self, dim: Union[int, List[int]], wires: int, index: Union[None, List[int]]
    ):
        self.index = index
        self.dim = dim
        self.wires = wires


class Circuit(nn.Module):
    def __init__(self, wires, dim=2, device="cpu"):
        super(Circuit, self).__init__()

        if isinstance(dim, int):
            self.dims_ = [dim] * wires
        elif isinstance(dim, list):
            if len(dim) != wires:
                raise ValueError(
                    f"Dimension list length {len(dim)} does not match number of wires {wires}."
                )
            self.dims_ = dim

        self.dim = dim
        self.width = int(np.prod(self.dims_))
        self.wires = wires
        self.device = device
        self.circuit = nn.Sequential()

        udits = sorted(list(set(self.dims_)))
        self.gates = [None] * (max(udits) + 1)
        for i in range(2, max(udits) + 1):
            if i in udits:
                self.gates[i] = GG.Gategen(dim=i, device=device)

        self.ops = []

    def make(self, *args, **kwargs):
        dim = -1
        if "dim" in kwargs:
            dim = kwargs["dim"]
            del kwargs["dim"]
        else:
            if isinstance(self.dim, int):
                dim = self.dim
            elif isinstance(self.dim, list):
                raise ValueError("Cannot auto-determine dimension from multiple wires.")

        return self.gates[dim].make(*args, **kwargs)

    def optimise(self):
        traced = torch.jit.trace(self, torch.randn(1, self.width, dtype=C64, device=self.device))
        return traced.eval()

    def gate(self, gate_or_name, index, **kwargs):
        if "device" not in kwargs:
            kwargs["device"] = self.device

        if isinstance(gate_or_name, GG.BaseGate):
            gate_instance = gate_or_name
            gate_instance.index = index
            gate_instance.wires = self.wires
            gate_instance.device = self.device
        elif callable(gate_or_name):
            if "device" not in kwargs:
                kwargs["device"] = self.device

            gate_instance = gate_or_name(
                dim=self.dim,
                wires=self.wires,
                index=index,
                **kwargs,
            )
        elif isinstance(gate_or_name, torch.Tensor):
            if gate_or_name.dim() != 2:
                raise ValueError("Tensor gate must be a 2D matrix.")
            gate_instance = GG.U(
                matrix=gate_or_name,
                dim=self.dim,
                wires=self.wires,
                index=index,
                device=self.device,
            )
        else:
            raise TypeError(f"Unsupported gate type: {type(gate_or_name)}")

        pos = str(len(self.circuit))
        self.circuit.add_module(pos, gate_instance)

    def matrix(self):
        basis = torch.zeros((self.width, 1), dtype=C64, device=self.device)

        columns = []
        for i in range(self.width):
            basis[i, 0] = 1.0
            columns.append(self.forward(basis))

        return torch.cat(columns, dim=1)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(dtype=C64, device=self.device)
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=C64, device=self.device)

        # if x.dim() == 1:
        #     x = x.view(-1, 1)
        # elif x.dim() > 2 or (x.dim() == 2 and x.shape[1] != 1):
        #     raise ValueError(f"Input state has an invalid shape: {x.shape}. Expected (total_dim, 1) or (total_dim,).")

        return self.circuit(x)