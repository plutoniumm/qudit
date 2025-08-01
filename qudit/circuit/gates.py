from typing import List, Optional, Union
from itertools import product
import torch.nn as nn
from math import log
import numpy as np
import torch

C64 = torch.complex64


def tensorise(m, device="cpu", dtype=C64):
    with torch.no_grad():
        if isinstance(m, torch.Tensor):
            return m
        elif isinstance(m, np.ndarray):
            return torch.from_numpy(m).to(device, non_blocking=True).type(dtype)
        elif isinstance(m, list):
            return torch.tensor(m, device=device, dtype=dtype)
        else:
            raise TypeError(
                f"Unsupported type for tensorisation: {type(m)}. Expected Tensor, ndarray, or list."
            )


def dec2den(dec, wires, dims):
    den = []
    temp_dec = dec
    for i in range(wires - 1, -1, -1):
        den.insert(0, temp_dec % dims[i])
        temp_dec //= dims[i]
    return den


def den2dec(den, dims):
    dec = 0
    for i in range(len(den)):
        stride = int(np.prod(dims[i + 1 :])) if i < len(den) - 1 else 1
        dec += den[i] * stride
    return dec


class BaseGate(nn.Module):
    def __init__(self, dim=2, index=[0], wires=1, inverse=False, device="cpu"):
        super().__init__()
        self.index = index
        self.device = device
        self.inverse = inverse
        self.wires = wires
        self.dims = [dim] * wires if isinstance(dim, int) else dim
        self.total_dim = int(np.prod(self.dims))

    # hijack ^ to do kron
    def __xor__(self, other):
        eye = torch.eye(self.total_dim, device=self.device)
        if isinstance(other, BaseGate):
            return self._getUnitary(eye, other.matrix())
        elif isinstance(other, torch.Tensor):
            return self._getUnitary(other, eye)
        else:
            raise TypeError(
                f"Unsupported type for kron operation: {type(other)}. Expected BaseGate or Tensor."
            )

    def _make_matrix(self, dim, fill_fn):
        return self._apply_inverse(fill_fn(dim))

    def _apply_inverse(self, M):
        return torch.conj(M).T.contiguous() if self.inverse else M

    def _infer_dits(self, x: torch.Tensor) -> int:
        dims: List[int] = self.dims

        same_dim = True
        for d in dims:
            if d != dims[0]:
                same_dim = False
                break

        if same_dim:
            return int(round(log(x.shape[0]) / log(dims[0])))
        return len(dims)

    def _getUnitary(self, x, gate_matrices):
        L = self._infer_dits(x)
        U = torch.eye(1, device=self.device)

        for i in range(L):
            if i in self.index:
                M = (
                    gate_matrices[i]
                    if isinstance(gate_matrices, dict)
                    else gate_matrices
                )
            else:
                M = torch.eye(self.dims[i], device=self.device)
            U = torch.kron(U, M)
        return U


class SingleDitGate(BaseGate):
    def __init__(
        self, dim=2, index=[0], wires=1, inverse=False, device="cpu", **kwargs
    ):
        super().__init__(dim, index, wires, inverse, device)
        self.M_dict = self._getMat(**kwargs)

    def _getMat(self, **kwargs):
        raise NotImplementedError

    def matrix(self):
        L = self.wires
        U = torch.eye(1, device=self.device)
        for i in range(L):
            if i in self.index:
                M = self.M_dict[i] if isinstance(self.M_dict, dict) else self.M_dict
            else:
                M = torch.eye(self.dims[i], device=self.device)
            U = torch.kron(U, M)
        return U


class H(SingleDitGate):
    def _getMat(self, **kwargs):
        M_dict = {}
        for idx in self.index:
            d = self.dims[idx]
            w = torch.tensor(np.exp(2 * 1j * np.pi / d), dtype=C64, device=self.device)
            M = torch.ones((d, d), dtype=C64, device=self.device)
            for i in range(1, d):
                for j in range(1, d):
                    M[i, j] = w ** (i * j)
            M = M / (d**0.5)
            M_dict[idx] = self._apply_inverse(M)
        return M_dict


class X(SingleDitGate):
    def _getMat(self, s=1, **kwargs):
        M_dict = {}
        for idx in self.index:
            d = self.dims[idx]

            def fill_fn(dim):
                M = torch.zeros((dim, dim), dtype=C64, device=self.device)
                for i in range(dim):
                    j = (i + s) % dim
                    M[j, i] = 1.0
                return M

            M_dict[idx] = self._make_matrix(d, fill_fn)
        return M_dict


class Z(SingleDitGate):
    def _getMat(self, s=1, **kwargs):
        M_dict = {}
        for idx in self.index:
            d = self.dims[idx]
            w = torch.tensor(np.exp(2 * 1j * np.pi / d), dtype=C64, device=self.device)

            def fill_fn(dim):
                return torch.diag(
                    torch.tensor(
                        [w ** (j * s) for j in range(dim)],
                        dtype=C64,
                        device=self.device,
                    )
                )

            M_dict[idx] = self._make_matrix(d, fill_fn)
        return M_dict


class Y(SingleDitGate):
    def _getMat(self, s=1, **kwargs):
        M_dict = {}
        dim = self.dims[0] if len(set(self.dims)) == 1 else self.dims
        x_gate = X(
            dim=dim,
            index=self.index,
            wires=self.wires,
            device=self.device,
            s=s,
        )
        z_gate = Z(
            dim=dim,
            index=self.index,
            wires=self.wires,
            device=self.device,
            s=s,
        )
        for i in self.index:
            M = torch.matmul(z_gate.M_dict[i], x_gate.M_dict[i]) / 1j
            M_dict[i] = M
        return M_dict


class ParametrizedRotation(BaseGate):
    def __init__(self, j=0, k=1, index=[0], dim=2, wires=1, device="cpu", angle=None):
        if isinstance(index, int):
            index = [index]

        super().__init__(dim, index, wires, False, device)
        assert angle is not None, "angle parameter is required and cannot be None"

        self.j_map = self._build_level_map(j)
        self.k_map = self._build_level_map(k) if k is not None else None

        if isinstance(angle, torch.Tensor):
            if angle.numel() != 1:
                raise ValueError("Angle tensor must contain a single scalar value.")
            self.angle = angle.to(device, dtype=torch.float32)
        elif isinstance(angle, (int, float)):
            self.angle = torch.tensor(angle, device=device, dtype=torch.float32)
        else:
            raise TypeError(
                f"Angle must be a single scalar (int or float), not {type(angle)}."
            )

    def _build_level_map(self, levels):
        if isinstance(levels, int):
            return {t: levels for t in self.index}
        if len(levels) != len(self.index):
            raise ValueError("Level list length must equal number of target dits")
        return {t: level for t, level in zip(self.index, levels)}

    def forward(self, x: torch.Tensor):
        psi = x.view(*self.dims)

        for i in self.index:
            # 1. Bring target to the front
            new_order = [i] + [j for j in range(self.wires) if j != i]
            psi_perm = psi.permute(*new_order).contiguous()

            # 2. Reshape
            d_i = self.dims[i]
            d_rest = self.total_dim // d_i
            psi_flat = psi_perm.view(d_i, d_rest)

            # 3. small (d_i x d_i) rotation mat
            j_ = self.j_map[i]
            k_ = self.k_map[i] if self.k_map is not None else 0
            M_block = self._getRMat(d_i, j_, k_, self.angle)

            # 4. small gate matrix
            psi_ = M_block @ psi_flat

            # 5. Reshape back
            psi_ = psi_.view(*[self.dims[j] for j in new_order])

            # 6. Permute
            inv_order = [new_order.index(j) for j in range(self.wires)]
            psi = psi_.permute(*inv_order).contiguous()

        # 7. Return the final
        return psi.view(self.total_dim, 1)



class RX(ParametrizedRotation):
    def __init__(self, index=[0], dim=2, wires=1, device="cpu", angle=None):
        super().__init__(0, 1, index, dim, wires, device, angle)

    def _getRMat(self, d: int, j: int, k: int, angle):
        M = torch.eye(d, device=self.device, dtype=C64)
        cos_hf = torch.cos(angle / 2)
        sin_hf_i = -1j * torch.sin(angle / 2)
        M[j, j] = cos_hf
        M[k, k] = cos_hf
        M[j, k] = sin_hf_i
        M[k, j] = sin_hf_i
        return M


class RY(ParametrizedRotation):
    def __init__(self, index=[0], dim=2, wires=1, device="cpu", angle=None):
        super().__init__(0, 1, index, dim, wires, device, angle)

    def _getRMat(self, d: int, j: int, k: int, angle_):
        M = torch.eye(d, device=self.device, dtype=C64)
        cos_hf = torch.cos(angle_ / 2)
        sin_hf = torch.sin(angle_ / 2)
        M[j, j] = cos_hf
        M[k, k] = cos_hf
        M[j, k] = -sin_hf
        M[k, j] = sin_hf
        return M


class RZ(ParametrizedRotation):
    def __init__(self, j=1, index=[0], dim=2, wires=1, device="cpu", angle=None):
        super().__init__(j, None, index, dim, wires, device, angle)

    def _getRMat(self, d: int, j: int, k: int, angle):
        if d == 2:
            phases = torch.empty(d, dtype=C64, device=self.device)
            phases[0] = torch.exp(1j * angle / 2)
            phases[1] = torch.exp(-1j * angle / 2)
            if j == 1:
                phases = phases.flip(dims=[0])
            M = torch.diag(phases)
            return M
        else:
            M = torch.eye(d, device=self.device, dtype=C64)
            phase = torch.exp(1j * angle)
            M[j, j] = phase

        return M


class GellMann:
    def __init__(self, j: int, k: int, d):
        self.j = j
        self.k = k
        self.d = d

        if self.j > self.k:
            t = "symm"
        elif self.k > self.j:
            t = "antisymm"
        elif self.j == self.k and self.j < self.d:
            t = "diag"
        else:
            t = "identity"

        self.type = t
        self.matrix = self._construct()

    def _construct(self):
        mat = np.zeros((self.d, self.d), dtype=np.complex64)

        if self.type == "symm":
            mat[self.j - 1, self.k - 1] = 1
            mat[self.k - 1, self.j - 1] = 1
        elif self.type == "antisymm":
            mat[self.j - 1, self.k - 1] = -1j
            mat[self.k - 1, self.j - 1] = 1j
        elif self.type == "diag":
            norm = np.sqrt(2 / (self.j * (self.j + 1)))
            for m in range(self.j):
                mat[m, m] = norm
            mat[self.j, self.j] = -self.j * norm
        else:
            np.fill_diagonal(mat, 1)

        return mat


def dGellMann(d):
    arr = [GellMann(j, k, d) for j in range(1, d + 1) for k in range(1, d + 1)]
    arr.reverse()
    return arr


class GMR(ParametrizedRotation):
    def __init__(
        self, j: int, k: int, index=[0], dim=2, wires=1, device="cpu", angle=None
    ):
        super().__init__(
            j=j, k=k, index=index, dim=dim, wires=wires, device=device, angle=angle
        )

    def _getRMat(self, d: int, j_, k_, angle_):
        gm = GellMann(j_ + 1, k_ + 1, d).matrix
        gm_tensor = torch.tensor(gm, dtype=C64, device=self.device)

        M = torch.eye(d, device=self.device, dtype=C64)
        c, s = torch.cos(angle_ / 2), torch.sin(angle_ / 2)
        M[j_, j_] = c
        M[k_, k_] = c
        M[j_, k_] = -1j * s * gm_tensor[j_, k_]
        M[k_, j_] = -1j * s * gm_tensor[k_, j_]
        return M


def T(dim=2, index=[0], wires=1, device="cpu"):
    w = torch.tensor(np.exp(2 * 1j * np.pi / (dim * 4)), dtype=C64, device=device)
    matrix = torch.diag(
        torch.tensor([w**j for j in range(dim)], dtype=C64, device=device)
    )

    return U(matrix=matrix, dim=dim, wires=wires, device=device, index=index)


def S(dim=2, index=[0], wires=1, device="cpu"):
    w = torch.tensor(np.exp(2 * 1j * np.pi / (dim * 2)), dtype=C64, device=device)
    matrix = torch.diag(
        torch.tensor([w**j for j in range(dim)], dtype=C64, device=device)
    )

    return U(matrix=matrix, dim=dim, wires=wires, device=device, index=index)


def P(theta, dim=2, index=[0], wires=1, device="cpu"):
    phases = [np.exp(1j * theta * j / (dim - 1)) for j in range(dim)]
    matrix = torch.diag(torch.tensor(phases, dtype=C64, device=device))

    return U(matrix=matrix, dim=dim, wires=wires, device=device, index=index)


class CU(BaseGate):
    def __init__(
        self,
        U_target,
        index=[0, 1],
        wires=2,
        dim=2,
        device="cpu",
        inverse=False,
    ):
        super().__init__(dim, index, wires, inverse, device)
        if len(self.index) != 2:
            raise ValueError(
                "Controlled gate requires exactly two indices: [control, target]."
            )

        self.control_idx = self.index[0]
        self.itarg = self.index[1]
        self.U_target = U_target.to(device)

        self.U = self._getCU()
        self.register_buffer("U_matrix", self._apply_inverse(self.U))

    def _getCU(self):
        L = torch.tensor(list(product(*[range(d) for d in self.dims]))).to(self.device)
        D = int(torch.prod(torch.tensor(self.dims)))

        U = torch.zeros((D, D), dtype=C64, device=self.device)

        for l in L:
            isrc = self._flat_index(l)
            if l[self.control_idx] == 1:
                for t in range(self.dims[self.itarg]):
                    l2 = l.clone()
                    l2[self.itarg] = t
                    idst = self._flat_index(l2)
                    U[idst, isrc] = self.U_target[t, l[self.itarg]]
            else:
                U[isrc, isrc] = 1.0 + 0j

        return U

    def _flat_index(self, state):
        idx = 0
        for i, d in enumerate(self.dims):
            stride = (
                int(torch.prod(torch.tensor(self.dims[i + 1 :])))
                if i + 1 < self.wires
                else 1
            )
            idx += state[i] * stride
        return idx

    def forward(self, x):
        return self.U_matrix @ x

    def matrix(self):
        return self.U_matrix

def pauli_x(d=2):
    if isinstance(d, list):
        d = d[0]
    X = torch.eye(d, dtype=C64)
    X = X.roll(1, dims=1)
    return X

def pauli_z(d=2):
    if isinstance(d, list):
        d = d[0]
    w = torch.exp(2j * torch.pi / d)
    return torch.diag(torch.tensor([w**i for i in range(d)], dtype=C64))


class CX(CU):
    def __init__(
        self, index=[0, 1], wires=2, dim=2, device="cpu", inverse=False
    ):
        sz = dim if isinstance(dim, int) else dim[index[1]]
        super().__init__(
            pauli_x(sz),
            index=index,
            wires=wires,
            dim=dim,
            device=device,
            inverse=inverse,
        )


class CZ(CU):
    def __init__(
        self, index=[0, 1], wires=2, dim=2, device="cpu", inverse=False
    ):
        sz = dim if isinstance(dim, int) else dim[index[1]]
        super().__init__(
            pauli_z(sz),
            index=index,
            wires=wires,
            dim=dim,
            device=device,
            inverse=inverse,
        )


class SWAP(BaseGate):
    def __init__(self, index=[0, 1], dim=2, wires=2, device="cpu"):
        super().__init__(dim, index, wires, False, device)
        if len(self.index) != 2:
            raise ValueError("SWAP gate requires exactly two index: [qudit1, qudit2].")
        self.q1 = self.index[0]
        self.q2 = self.index[1]
        self.U = self._getMat()
        self.register_buffer("U_matrix", self.U)

    def _getMat(self):
        U = torch.zeros((self.total_dim, self.total_dim), device=self.device, dtype=C64)
        for k in range(self.total_dim):
            localr = dec2den(k, self.wires, self.dims)
            locall = list(localr)

            locall[self.q1], locall[self.q2] = (
                localr[self.q2],
                localr[self.q1],
            )

            globall = den2dec(locall, self.dims)
            U[globall, k] = 1
        return U

    def forward(self, x):
        return self.U_matrix @ x

    def matrix(self):
        return self.U_matrix


class U(BaseGate):
    def __init__(self, matrix=None, dim=2, wires=1, device="cpu", index=None):
        super().__init__(
            dim,
            index if index is not None else range(wires),
            wires,
            False,
            device,
        )
        if index is None:
            self.index = None
        elif isinstance(index, int):
            self.index = [index]
        else:
            self.index = list(index)
            self.index.sort()

        if self.index is None:
            self.sub_dims = self.dims
            self.sub_dim = self.total_dim
        else:
            self.sub_dims = [self.dims[i] for i in self.index]
            self.sub_dim = int(np.prod(self.sub_dims))

        self.M = tensorise(matrix, device=device, dtype=C64)
        if self.M.shape != (self.sub_dim, self.sub_dim):
            raise ValueError(
                f"Provided matrix dimensions ({self.M.shape}) do not match the product of the targeted qudits' dimensions ({self.sub_dim})."
            )

    def forward(self, x):
        if self.index is None:
            return self.M @ x
        else:
            all_index = list(range(self.wires))
            itarg = self.index
            unused = [i for i in all_index if i not in itarg]
            new_order = itarg + unused
            inv_order = [new_order.index(i) for i in range(self.wires)]

            psi = x.view(*self.dims)
            psi_perm = psi.permute(*new_order).contiguous()

            d_rest = self.total_dim // self.sub_dim
            psi_flat = psi_perm.reshape(self.sub_dim, d_rest)

            psi_ = self.M @ psi_flat

            new_shape = [self.dims[i] for i in new_order]
            psi_ = psi_.reshape(*new_shape)
            psi_final = psi_.permute(*inv_order).contiguous()
            return psi_final.view(self.total_dim, 1)

    def matrix(self):
        if self.index is None:
            return self.M

        basis_index = list(product(*[range(d) for d in self.dims]))
        perm = []
        all = list(range(self.wires))
        itarg = self.index
        unused = [i for i in all if i not in itarg]
        new_order = itarg + unused
        new_dims = [self.dims[i] for i in new_order]

        for m in basis_index:
            m_list = list(m)
            permuted = [m_list[i] for i in new_order]
            new_dec = den2dec(permuted, new_dims)
            perm.append(new_dec)

        perm = torch.tensor(perm, dtype=torch.long, device=self.device)
        P = torch.zeros((self.total_dim, self.total_dim), dtype=C64, device=self.device)
        for i in range(self.total_dim):
            P[i, perm[i]] = 1.0

        d_rest = self.total_dim // self.sub_dim
        I_ = torch.eye(d_rest, dtype=C64, device=self.device)

        return P.T @ torch.kron(self.M, I_) @ P


class Gategen:
    def __init__(self, dim=2, device="cpu"):
        self.dim = dim
        self.device = device

    @property
    def I(self):
        return torch.eye(self.dim, dtype=C64, device=self.device)

    @property
    def H(self):
        h_gate = H(dim=self.dim, index=[0], wires=1, device=self.device)
        m = h_gate.M_dict[0]
        m.__name__ = "H"
        return m

    @property
    def X(self):
        x_gate = X(dim=self.dim, index=[0], wires=1, device=self.device)
        m = x_gate.M_dict[0]
        m.__name__ = "X"
        return m

    @property
    def Z(self):
        z_gate = Z(dim=self.dim, index=[0], wires=1, device=self.device)
        m = z_gate.M_dict[0]
        m.__name__ = "Z"
        return m

    @property
    def Y(self):
        y_gate = Y(dim=self.dim, index=[0], wires=1, device=self.device)
        m = y_gate.M_dict[0]
        m.__name__ = "Y"
        return m

    @property
    def S(self):
        w = torch.tensor(
            np.exp(2 * 1j * np.pi / (self.dim * 2)), dtype=C64, device=self.device
        )
        m = torch.diag(
            torch.tensor([w**j for j in range(self.dim)], dtype=C64, device=self.device)
        )
        m.__name__ = "S"
        return m

    @property
    def T(self):
        w = torch.tensor(
            np.exp(2 * 1j * np.pi / (self.dim * 4)), dtype=C64, device=self.device
        )
        m = torch.diag(
            torch.tensor([w**j for j in range(self.dim)], dtype=C64, device=self.device)
        )
        m.__name__ = "T"
        return m

    def P(self, theta):
        phases = [np.exp(1j * theta * j / (self.dim - 1)) for j in range(self.dim)]
        m = torch.diag(torch.tensor(phases, dtype=C64, device=self.device))
        m.__name__ = "P"
        return m

    def RX(self, *args, **kwargs):
        if "device" in kwargs:
            return RX(*args, **kwargs)
        else:
            angle = args[0] if len(args) > 0 else kwargs.get("angle", None)
            print(args, kwargs)

            rx_gate = RX(
                index=[0], dim=self.dim, wires=1, device=self.device, angle=angle
            )
            m = rx_gate._getRMat(self.dim, j=0, k=1, angle=angle)
            m.__name__ = "RX"
            return m

    def RY(self, *args, **kwargs):
        if "device" in kwargs:
            return RY(*args, **kwargs)
        else:
            angle = args[0] if len(args) > 0 else kwargs.get("angle", None)

        ry_gate = RY(index=[0], dim=self.dim, wires=1, device=self.device, angle=angle)
        m = ry_gate._getRMat(self.dim, j=0, k=1, angle=angle)
        m.__name__ = "RY"
        return m

    def RZ(self, *args, **kwargs):
        if "device" in kwargs:
            return RZ(*args, **kwargs)
        else:
            angle = args[0] if len(args) > 0 else kwargs.get("angle", None)

        rz_gate = RZ(
            j=1, index=[0], dim=self.dim, wires=1, device=self.device, angle=angle
        )
        m = rz_gate._getRMat(self.dim, j=1, k=0, angle=angle)
        m.__name__ = "RZ"
        return m

    @property
    def CX(self):
        cx_gate = CX(dim=self.dim, index=[0, 1], wires=2, device=self.device)
        cx_gate = cx_gate.matrix()
        cx_gate.__name__ = "CX"
        return cx_gate

    @property
    def CZ(self):
        cz_gate = CZ(dim=self.dim, index=[0, 1], wires=2, device=self.device)
        m = cz_gate.matrix()
        m.__name__ = "CZ"
        return m

    @property
    def SWAP(self):
        swap_gate = SWAP(dim=self.dim, index=[0, 1], wires=2, device=self.device)
        m = swap_gate.matrix()
        m.__name__ = "SWAP"
        return m

    def make(self, matrix):
        def gate_func(dim=2, wires=1, index=None, **kwargs):
            if index is None:
                raise ValueError("index parameter is required")

            if isinstance(dim, int):
                matrix_shape = np.array(matrix).shape[0]
                num_qudits = int(round(np.log(matrix_shape) / np.log(dim)))
                if dim**num_qudits != matrix_shape:
                    raise ValueError(
                        f"Matrix dimension {matrix_shape} doesn't match dim^n for any integer n"
                    )
                index = list(range(num_qudits))
            else:
                index = list(range(len(dim)))

            if isinstance(matrix, torch.Tensor):
                pmatrix = matrix.to(self.device, non_blocking=True)
            else:
                pmatrix = torch.tensor(matrix, device=self.device, dtype=C64)

            pmatrix = pmatrix.to_sparse()
            return U(
                matrix=pmatrix,
                dim=dim,
                wires=wires,
                device=self.device,
                index=index,
            )

        gate_func.__name__ = "GG.make"
        return gate_func
