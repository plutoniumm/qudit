from torch import nn, tensor, complex64 as Cplx, Tensor, from_numpy, jit, randn
from dataclasses import dataclass
from typing import Union, List
from . import gates as GG
import numpy as np

@dataclass
class Gateless:
    index: Union[None, List[int]]
    dim: Union[int, List[int]]
    wires: int

    def __init__(self, dim: Union[int, List[int]], wires: int, index: Union[None, List[int]]):
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
        self.gates = [None] * (max(udits) - 1)
        for i in range(2, max(udits) + 1):
            if i in udits:
                self.gates[i-2] = GG.Gategen(dim=i, device=device)

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

        return self.gates[dim - 2].make(*args, **kwargs)

    def optimise(self):
        traced = jit.trace(self, randn(1, self.width, dtype=Cplx, device=self.device))
        return traced.eval()

    def gate(self, gate_or_name, indices, **kwargs):
        pos = str(len(self.circuit))
        if "device" not in kwargs:
            kwargs["device"] = self.device

        if callable(gate_or_name):
            gate_instance = gate_or_name(
                dim=self.dim,
                wires=self.wires,
                index=indices,
                **kwargs,
            )
        elif isinstance(gate_or_name, GG.BaseGate):
            gate_instance = gate_or_name
            gate_instance.wires = self.wires
            gate_instance.device = self.device
            gate_instance.index = indices
        else:
            raise ValueError(f"Unsupported gate type: {type(gate_or_name)}. ")

        self.circuit.add_module(pos, gate_instance)

    def forward(self, x):
        if isinstance(x, Tensor):
            return self.circuit(x)

        if isinstance(x, np.ndarray):
            x = from_numpy(x).to(dtype=Cplx, device=self.device)
        else:
            x = tensor(x, dtype=Cplx, device=self.device)

        return self.circuit(x)
