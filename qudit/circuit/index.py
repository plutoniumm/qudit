from .gates import Unitary, Gategen, Operator
from typing import Union as U, List, Any, Optional, Dict
from dataclasses import dataclass
import torch.nn as nn
from enum import Enum
import numpy as np
import torch

C64 = torch.complex64
Array = List[int]


class Mode(Enum):
    """
    Execution mode for the circuit: statevector (VECTOR) or density-matrix (MATRIX).
    """

    VECTOR = "vector"
    MATRIX = "matrix"


@dataclass
class Frame:
    """
    Lightweight record of a circuit operation (gate name, target indices, local dims, and parameters).
    """

    name: str
    index: List[int]
    dim: U[int, Array]
    params: Dict[str, Any]

    def __init__(
        self,
        dim: U[int, Array],
        index: List[int],
        name: str,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Create an operation frame describing a gate acting on `index` with given local `dim` and `params`.
        """
        self.index = index
        self.dim = dim
        self.name = name
        self.params = params or {}

    @staticmethod
    def parse(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract a small, human-readable parameter subset (e.g. angles/indices) for display/logging.
        """
        valid = ["i", "j", "k", "angle", "type"]
        params: Dict[str, Any] = {}

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
    def create(
        gate_in: Any, dims: List[int], index: List[int], **kwargs: Any
    ) -> "Frame":
        """
        Create a Frame from a gate-like input, inferring a display name and capturing key params.
        """
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
        """
        Pretty-print as `NAME(k=v, ...)` for circuit diagrams.
        """
        if self.params:
            items = ", ".join(f"{k}={self.params[k]}" for k in sorted(self.params))

            return f"{self.name}({items})"
        return f"{self.name}"


class Circuit(nn.Module):
    """
    A qudit circuit as an `nn.Module` applying a sequence of embedded unitaries. In VECTOR (default) mode it applies

    $|\psi\\rangle \mapsto U_m \cdots U_2 U_1 |\psi\\rangle$,

    whereas in MATRIX mode it applies the unitary channel,

    $\\rho \mapsto U_m \cdots U_2 U_1\, \\rho\, U_1^\dagger U_2^\dagger \cdots U_m^\dagger$.
    """

    mode: Mode
    dims_: List[int]
    dim: U[int, Array]
    width: int
    wires: int
    device: str
    circuit: nn.Sequential
    operations: List[Frame]
    gates: Dict[int, Gategen]
    gate_gen: Any

    def __init__(
        self,
        wires: int = 2,
        dim: U[int, Array] = 2,
        device: str = "cpu",
        mode: U[Mode, str] = Mode.VECTOR,
    ):
        """
        Initialize a circuit with `wires` subsystems of local dimension(s) `dim` on `device`.
        """
        super(Circuit, self).__init__()

        if isinstance(mode, str):
            mode = mode.lower()
        self.mode = Mode(mode)

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
        self.operations = []

        udits = sorted(list(set(self.dims_)))
        self.gates = {}
        for d in udits:
            self.gates[d] = Gategen(dim=d, device=device)

        if isinstance(self.dim, int):
            self.gate_gen = self.gates[self.dim]

    def gate(self, gate_in: Any, index: Any, **kwargs: Any) -> None:
        """
        Append a gate acting on `index` to the circuit with optionally params

        Operator may be of type Unitary, Operator, torch.Tensor, or a callable factory for an embedded Unitary.

        The gate is added to the circuit as an nn.Module and a Frame is recorded for display/logging.
        """
        if "device" not in kwargs:
            kwargs["device"] = self.device

        idx_list = index if isinstance(index, list) else [index]
        dims = [self.dims_[i] for i in idx_list]

        self.operations.append(Frame.create(gate_in, dims, idx_list, **kwargs))

        Instance: Any = None
        if isinstance(gate_in, (torch.Tensor, Operator)):
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

    def forward(self, x: Any) -> torch.Tensor:
        """
        Apply the circuit to a statevector or density matrix depending on `self.mode`.
        """
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

            for module in self.circuit:  # type: ignore[assignment]
                rho = module.forwardd(rho)  # type: ignore[attr-defined]
            return rho

    def matrix(self) -> torch.Tensor:
        """
        Materialize the full unitary matrix by acting on the computational basis vectors.
        """
        W = self.width
        I = torch.eye(W, dtype=C64, device=self.device)
        cols = []
        for i in range(W):
            cols.append(self.circuit(I[i]))
        return torch.cat(cols, dim=1)

    def draw(self, mode: str = "ascii") -> Any:
        """
        Render a simple textual diagram of the circuit; currently supports `mode='ascii'`.
        """
        from .transform import Table

        table = Table(self)

        if mode == "ascii":
            return table.draw(self)

        return table.draw(self)
