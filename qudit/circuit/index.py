from .gates import Unitary, Gategen, Gate
from typing import Union as U, List
from dataclasses import dataclass
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
class Frame:
    name: str
    index: List[int]
    dim: U[int, Array]
    params: dict

    def __init__(
        self,
        dim: U[int, Array],
        index: List[int],
        name: str,
        params: dict = {},
    ):
        self.index = index
        self.dim = dim
        self.name = name
        self.params = params

    @staticmethod
    def parse(kwargs) -> dict:
        valid = ["i", "j", "k", "angle", "type"]
        params = {}

        for key in valid:
            if key in kwargs:
                val = kwargs[key]
                if isinstance(val, torch.Tensor):
                    val = val.item()

                if isinstance(val, float):
                    val = round(val, 4)

                params[key] = val

        return params

    @staticmethod
    def create(gate_in, dims, index, **kwargs) -> "Frame":
        params = Frame.parse(kwargs)

        name = ""
        if hasattr(gate_in, "name"):
            name = gate_in.name
        elif hasattr(gate_in, "__name__"):
            name = gate_in.__name__
        else:
            name = "U"

        return Frame(dim=dims, index=index, name=name, params=params)

    def __str__(self) -> str:
        if self.params:
            items = ", ".join(
                f"{k}={self.params[k]}"
                    for k in sorted(self.params)
            )

            return f"{self.name}({items})"
        return f"{self.name}"


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
        self.operations: List[Frame] = []

        udits = sorted(list(set(self.dims_)))
        self.gates = {}
        for d in udits:
            self.gates[d] = Gategen(dim=d, device=device)

        if isinstance(self.dim, int):
            self.gate_gen = self.gates[self.dim]

    def gate(self, gate_in, index, **kwargs):
        if "device" not in kwargs:
            kwargs["device"] = self.device

        idx_list = index if isinstance(index, list) else [index]
        dims = [self.dims_[i] for i in idx_list]

        self.operations.append(Frame.create(gate_in, dims, idx_list, **kwargs))

        Instance = None
        if isinstance(gate_in, (torch.Tensor, Gate)):
            Instance = Unitary(
                matrix=gate_in,
                index=idx_list,
                wires=self.wires,
                dim=self.dims_,
                device=self.device,
            )
        elif callable(gate_in):
            Instance = gate_in(
                dim=self.dims_, wires=self.wires, index=idx_list, **kwargs
            )
        else:
            raise TypeError(f"Unsupported gate input: {type(gate_in)}")

        pos = str(len(self.circuit))
        self.circuit.add_module(pos, Instance)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(dtype=C64, device=self.device)
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=C64, device=self.device)
        else:
            x = x.to(dtype=C64, device=self.device)

        if self.mode == Mode.VECTOR:
            return self.circuit(x)
        else:  # Density matrix logic
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
        W = self.width
        I = torch.eye(W, dtype=C64, device=self.device)
        cols = []
        for i in range(W):
            cols.append(self.circuit(I[i]))
        return torch.cat(cols, dim=1)

    def draw(self, mode="ascii"):
        from .transform import Table
        table = Table(self)

        if mode == "ascii":
            return table.draw(self)