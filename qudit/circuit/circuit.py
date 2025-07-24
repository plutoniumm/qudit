from torch import nn, tensor, complex64 as Cplx, Tensor, from_numpy
from . import gates as GG

# import gates as GG
import numpy as np


class Circuit(nn.Module):
    def __init__(self, dits, dim=2, device="cpu"):
        super(Circuit, self).__init__()

        self.dim = dim
        self.dits = dits
        self.device = device
        self.circuit = nn.Sequential()
        self.gates = GG.Gategen(dim=dim, device=device)

    def make(self, *args, **kwargs):
        return self.gates.make(*args, **kwargs)

    def gate(self, gate_or_name, indices, **kwargs):
        pos = str(len(self.circuit))

        if isinstance(gate_or_name, Tensor):
            if hasattr(gate_or_name, "name"):
                name = gate_or_name.name
            else:
                name = None

            gate_or_name = gate_or_name.to_sparse_coo()
            gate_or_name = self.gates.make(gate_or_name, name)

        if callable(gate_or_name):
            gate_instance = gate_or_name(
                dim=self.dim,
                dits=self.dits,
                index=indices,
                **kwargs,
            )
        elif isinstance(gate_or_name, str):
            if not hasattr(GG, gate_or_name):
                raise ValueError(
                    f"Gate '{gate_or_name}' is not defined in the gates module."
                )

            mod = getattr(GG, gate_or_name)
            gate_instance = mod(
                dim=self.dim,
                dits=self.dits,
                device=self.device,
                index=indices,
                **kwargs,
            )
        elif isinstance(gate_or_name, GG.BaseGate):
            gate_instance = gate_or_name
            gate_instance.dim = self.dim
            gate_instance.dits = self.dits
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
