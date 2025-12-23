from dataclasses import dataclass, field
from typing import Union as U, List, Tuple
from . import gates as GG
import torch.nn as nn
from enum import Enum
import numpy as np
import torch

C64 = torch.complex64
Array = List[int]

class Mode(Enum):
    VECTOR = "vector"
    MATRIX = "matrix"


@dataclass
class Gateless:
    index: U[None, Array]
    dim: U[int, Array]
    wires: int

    def __init__(
        self, dim: U[int, Array], wires: int, index: U[None, Array]
    ):
        self.index = index
        self.dim = dim
        self.wires = wires


class Circuit(nn.Module):
    def __init__(
            self,
            wires: int = 2,
            dim: U[int, Array] = 2,
            device: str = "cpu",
            mode: U[Mode, str] = Mode.VECTOR,
        ):
        super(Circuit, self).__init__()

        if isinstance(mode, str):
            mode = mode.lower()
        self.mode: Mode = Mode(mode)

        if isinstance(dim, int):
            self.dims_ = [dim] * wires
        elif isinstance(dim, list):
            if len(dim) != wires:
                raise ValueError(f"Dim list {len(dim)} != wires {wires}.")
            self.dims_ = dim

        self.dim = dim
        self.width = int(np.prod(self.dims_))
        self.wires = wires
        self.device = device

        self.circuit = nn.Sequential()
        self.operations: List[Tuple[str, List[int], List[int]]] = []

        udits = sorted(list(set(self.dims_)))
        self.gates = {}
        for d in udits:
            self.gates[d] = GG.Gategen(dim=d, device=device)

        if isinstance(self.dim, int):
             self.gate_gen = self.gates[self.dim]

    def gate(self, gate_in, index, **kwargs):
        if "device" not in kwargs:
            kwargs["device"] = self.device

        idx_list = index if isinstance(index, list) else [index]

        # g_name = gate_in.__name__ or "Unknown"
        gate_instance = None

        if isinstance(gate_in, torch.Tensor):
            gate_instance = GG.Unitary(
                matrix=gate_in,
                index=idx_list,
                wires=self.wires,
                dim=self.dims_,
                device=self.device,
                # name=g_name
            )

        elif callable(gate_in):
            # print(gate_in)
            gate_instance = gate_in(
                dim=self.dims_,
                wires=self.wires,
                index=idx_list,
                **kwargs
            )
            # g_name = gate_instance.__name__

        else:
             raise TypeError(f"Unsupported gate input: {type(gate_in)}")

        # Add to Sequential
        pos = str(len(self.circuit))
        self.circuit.add_module(pos, gate_instance)

        target_dims = [self.dims_[i] for i in idx_list]
        # self.operations.append((g_name, idx_list, target_dims))

    def forward(self, x):
        # Input standardisation
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(dtype=C64, device=self.device)
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=C64, device=self.device)
        else:
            x = x.to(dtype=C64, device=self.device)

        if self.mode == Mode.VECTOR:
            return self.circuit(x)
        else: # Density matrix logic
            W = self.width
            if x.dim() == 1 or (x.dim() == 2 and min(x.shape) == 1):
                psi = x.reshape(W, 1)
                rho = psi @ psi.conj().T
            else:
                rho = x

            for module in self.circuit:
                rho = module.forwardd(rho)
            return rho

    def matrix(self):
        """Return the Unitary of the whole circuit"""
        W = self.width
        I = torch.eye(W, dtype=C64, device=self.device)
        cols = []
        for i in range(W):
            cols.append(self.circuit(I[i]))
        return torch.cat(cols, dim=1)