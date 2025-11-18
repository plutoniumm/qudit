from dataclasses import dataclass
from typing import Union as U, List
from . import gates as GG
import torch.nn as nn
from enum import Enum
import numpy as np
import torch

""""
Most of this file is just 1 line

circuit = nn.Sequential()

Everything else is
    ifelse gom jabbar
just to make and cases types work.
"""

C64, N64 = torch.complex64, np.complex64
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
        W = self.width
        I = torch.eye(W, dtype=C64, device=self.device)

        init = torch.zeros((W, W), dtype=C64, device=self.device)
        for i in range(W):
            v = self._apply_vector(I[i])
            if isinstance(v, torch.Tensor) and v.dim() == 2:
                init[i, :] = v.T[0]
            else:
                init[i, :] = v

        return init

    def _apply_vector(self, x: torch.Tensor):
        return self.circuit(x)

    def _apply_matrix(self, out: torch.Tensor):
        for module in self.circuit:
            out = module.forwardd(out)

        return out

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(dtype=C64, device=self.device)
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=C64, device=self.device)
        else:
            x = x.to(dtype=C64, device=self.device)

        if self.mode == Mode.VECTOR:
            return self._apply_vector(x)
        else:
            W = self.width
            if x.dim() == 1 or (x.dim() == 2 and min(x.shape) == 1):
                psi = x.reshape(W, -1)
                if psi.shape[1] != 1:
                    psi = psi.view(W, 1)
                rho = psi @ torch.conj(psi).T
            else:
                if x.dim() != 2 or x.shape[0] != W or x.shape[1] != W:
                    raise ValueError(
                        f"In matrix mode, input must be a (W x W) density matrix or a length-W vector. Got shape {tuple(x.shape)} with W={W}."
                    )
                rho = x
            return self._apply_matrix(rho)
