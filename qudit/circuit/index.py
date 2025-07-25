from torch import nn, tensor, complex64 as Cplx, Tensor, from_numpy
from . import gates as GG
import numpy as np


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
        self.gates = GG.Gategen(dim=dim, device=device)

    def make(self, *args, **kwargs):
        return self.gates.make(*args, **kwargs)

    def gate(self, gate_or_name, indices, **kwargs):
        pos = str(len(self.circuit))
        if "device" not in kwargs:
            kwargs["device"] = self.device

        if isinstance(gate_or_name, Tensor):
            gate_or_name = gate_or_name.to_sparse_coo()
            gate_or_name = self.gates.make(gate_or_name)

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
