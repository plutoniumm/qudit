from typing import List, Optional, Union, Callable
import torch.nn as nn
import numpy as np
import torch

C64 = torch.complex64


class Gate:
    def __init__(self, tensor: torch.Tensor, name: str, params: List = []):
        self.tensor = tensor
        self.name = name
        self.params = params

    def __getattr__(self, item):
        return getattr(self.tensor, item)

    def __repr__(self):
        if self.params:
            return f"{self.name!r}({self.params})"
        else:
            return self.name


def tensorise(m, device="cpu", dtype=C64):
    if isinstance(m, torch.Tensor):
        return m.to(device=device, dtype=dtype)
    elif isinstance(m, np.ndarray):
        return torch.from_numpy(m).to(device, non_blocking=True).type(dtype)
    elif isinstance(m, list):
        return torch.tensor(m, device=device, dtype=dtype)
    elif isinstance(m, Gate):
        return m.tensor.to(device=device, dtype=dtype)
    else:
        raise TypeError(
            f"Unsupported type: {type(m)}. Expected Tensor, ndarray, or list."
        )


def gell_mann(j: int, k: int, d: int, device="cpu"):
    m = torch.zeros((d, d), dtype=C64, device=device)

    if j < k:
        m[j, k] = 1.0
        m[k, j] = 1.0
    elif j > k:
        m[k, j] = torch.tensor(-1j, dtype=C64, device=device)
        m[j, k] = torch.tensor(1j, dtype=C64, device=device)
    else:
        l = j + 1
        if l >= d:
            return torch.eye(d, dtype=C64, device=device)

        scale = np.sqrt(2 / (l * (l + 1)))
        for i in range(l):
            m[i, i] = scale
        m[l, l] = -l * scale

    return m


class Unitary(nn.Module):
    def __init__(
        self,
        matrix,
        index: List[int],
        wires: int,
        dim: Union[int, List[int]],
        device="cpu",
        name="U",
        params: Optional[list] = None,
    ):
        super().__init__()
        self.device = device
        self.wires = wires
        self.index = index if isinstance(index, list) else [index]
        self.dims = [dim] * wires if isinstance(dim, int) else dim
        self.name = name
        self.params = list(params) if params is not None else []

        self.total_dim = int(np.prod(self.dims))
        self.target_dims = [self.dims[i] for i in self.index]
        self.target_size = int(np.prod(self.target_dims))

        self.U = tensorise(matrix, device=device)
        if self.U.shape != (self.target_size, self.target_size):
            raise ValueError(
                f"Matrix shape {self.U.shape} does not match target size {(self.target_size, self.target_size)}."
            )

        self.all = list(range(self.wires))
        self.unused = [i for i in self.all if i not in self.index]
        self.perm = self.index + self.unused

        self.inv_perm = [self.perm.index(i) for i in range(self.wires)]

        self.rest_size = self.total_dim // self.target_size

    def forward(self, x: torch.Tensor):
        psi = x.view(*self.dims)
        psi = psi.permute(*self.perm)
        psi_flat = psi.reshape(self.target_size, self.rest_size)
        psi_out = self.U @ psi_flat
        current_dims = [self.dims[i] for i in self.perm]
        psi_out = psi_out.view(*current_dims)
        psi_final = psi_out.permute(*self.inv_perm).contiguous()
        return psi_final.view(self.total_dim, 1)

    def forwardd(self, rho: torch.Tensor):
        U = self.matrix()
        return U @ rho @ U.conj().T

    def matrix(self):
        eye = torch.eye(self.total_dim, device=self.device, dtype=C64)
        cols = []
        for i in range(self.total_dim):
            cols.append(self.forward(eye[i]))
        return torch.cat(cols, dim=1)


class Gategen:
    def __init__(self, dim=2, device="cpu"):
        self.dim = dim
        self.device = device

    def asU(
        self,
        m: torch.Tensor,
        index,
        wires: int,
        dim: Union[int, List[int]],
        name: Optional[str] = None,
        params: Optional[list] = None,
    ):
        return Unitary(
            m,
            index=index,
            wires=wires,
            dim=dim,
            device=self.device,
            name=name or "U",
            params=params,
        )

    @property
    def I(self):
        return Gate(torch.eye(self.dim, dtype=C64, device=self.device), "I")

    @property
    def H(self):
        d = self.dim
        if d == 2:
            m = torch.tensor(
                [[1, 1], [1, -1]], dtype=C64, device=self.device
            ) / np.sqrt(2)
        else:
            w = np.exp(2j * torch.pi / d)
            idx = torch.arange(d, device=self.device)
            m = (w ** torch.outer(idx, idx)) / np.sqrt(d)
        return Gate(m.to(dtype=C64), "H")

    @property
    def X(self):
        d = self.dim
        if d == 2:
            m = torch.tensor([[0, 1], [1, 0]], dtype=C64, device=self.device)
        else:
            m = torch.roll(
                torch.eye(d, dtype=C64, device=self.device), shifts=1, dims=1
            )
        return Gate(m, "X")

    @property
    def Z(self):
        d = self.dim
        if d == 2:
            m = torch.tensor([[1, 0], [0, -1]], dtype=C64, device=self.device)
        else:
            w = np.exp(2j * torch.pi / d)
            idx = torch.arange(d, device=self.device)
            m = torch.diag(w**idx)
        return Gate(m, "Z")

    @property
    def Y(self):
        d = self.dim
        if d == 2:
            m = torch.tensor([[0, -1j], [1j, 0]], dtype=C64, device=self.device)
        else:
            m = torch.matmul(self.Z.tensor, self.X.tensor) / 1j
        return Gate(m, "Y")

    def GMR(self, j, k, angle, type="asym", *, matrix: bool = False, **kwargs):
        if not isinstance(angle, torch.Tensor):
            angle = torch.tensor(angle, dtype=C64, device=self.device)

        if type == "sym":
            idx1, idx2 = min(j, k), max(j, k)
            if idx1 == idx2:
                raise ValueError("Symmetric requires distinct j, k")
        elif type == "asym":
            idx1, idx2 = max(j, k), min(j, k)
        elif type == "diag":
            idx1, idx2 = j, j
        else:
            raise ValueError("type must be sym, asym, or diag")

        gen = gell_mann(idx1, idx2, self.dim, device=self.device)

        if type in ["sym", "asym"]:
            m = torch.eye(self.dim, dtype=C64, device=self.device)
            c = torch.cos(angle / 2)
            s = torch.sin(angle / 2)

            a, b = min(j, k), max(j, k)

            m[a, a] = c
            m[b, b] = c

            if type == "sym":
                m[a, b] = -1j * s
                m[b, a] = -1j * s
            else:
                m[a, b] = -s
                m[b, a] = s

            gate_name = f"GMR_{type}"
        else:
            m = torch.matrix_exp(-1j * (angle / 2) * gen)
            gate_name = f"GMR_{type}"

        gate_params = [
            ("type", type),
            ("j", j),
            ("k", k),
            ("angle", angle),
            ("dim", self.dim),
        ]

        if matrix:
            return Gate(m, gate_name, params=gate_params)

        index = kwargs.pop("index")
        wires = kwargs.pop("wires")
        dim = kwargs.pop("dim")
        name = kwargs.pop("name", None)
        return self.asU(
            m,
            index=index,
            wires=wires,
            dim=dim,
            name=name or gate_name,
            params=gate_params,
        )

    def RX(self, angle, *, matrix: bool = False, **kwargs):
        if matrix:
            return self.GMR(0, 1, angle, type="sym", matrix=True)
        return self.GMR(0, 1, angle, type="sym", **kwargs)

    def RY(self, angle, *, matrix: bool = False, **kwargs):
        if matrix:
            return self.GMR(0, 1, angle, type="asym", matrix=True)
        return self.GMR(0, 1, angle, type="asym", **kwargs)

    def RZ(self, angle, *, matrix: bool = False, **kwargs):
        if matrix:
            return self.GMR(0, 0, angle, type="diag", matrix=True)
        return self.GMR(0, 0, angle, type="diag", **kwargs)

    def CU(self, U_target=None, *, matrix: bool = False, **kwargs):
        d = self.dim
        ctrl_state = 1

        blocks = [torch.eye(d, device=self.device, dtype=C64) for _ in range(d)]
        blocks[ctrl_state] = tensorise(
            U_target.tensor if isinstance(U_target, Gate) else U_target,
            device=self.device,
        )

        m = torch.block_diag(*blocks)
        gate_name = "CU"

        target_name = U_target.name if isinstance(U_target, Gate) else None
        gate_params = [
            ("ctrl_state", ctrl_state),
            ("target", target_name),
            ("dim", d),
        ]

        if matrix:
            return Gate(m, gate_name, params=gate_params)

        index = kwargs.pop("index")
        wires = kwargs.pop("wires")
        dim = kwargs.pop("dim")
        name = kwargs.pop("name", None)
        return self.asU(
            m,
            index=index,
            wires=wires,
            dim=dim,
            name=name or gate_name,
            params=gate_params,
        )

    @property
    def CX(self):
        return self.CU(self.X, matrix=True)

    def CX_gate(self, *, matrix: bool = False, **kwargs):
        if matrix:
            return self.CX
        return self.CU(self.X, **kwargs)

    @property
    def SWAP(self):
        d = self.dim
        m = torch.zeros((d * d, d * d), dtype=C64, device=self.device)
        for i in range(d):
            for j in range(d):
                row = i * d + j
                col = j * d + i
                m[col, row] = 1.0
        return Gate(m, "SWAP")

    def SWAP_gate(self, *, matrix: bool = False, **kwargs):
        if matrix:
            return self.SWAP
        index = kwargs.pop("index")
        wires = kwargs.pop("wires")
        dim = kwargs.pop("dim")
        name = kwargs.pop("name", None)

        m = self.SWAP.tensor
        return self.asU(
            m,
            index=index,
            wires=wires,
            dim=dim,
            name=name or "SWAP",
            params=[("dim", self.dim)],
        )

    def make(self, matrix):
        t = tensorise(matrix, device=self.device)

        def factory(dim, wires, index, **kwargs):
            name = kwargs.get("name") or "Custom"
            params = kwargs.get("params")
            return Unitary(
                t, index, wires, dim, device=self.device, name=name, params=params
            )

        return factory
