from itertools import product
from . import utils as Ut
import torch.nn as nn
from math import log
import numpy as np
import torch

Cplx = torch.complex64


def eye(dim, device):
    return torch.sparse_coo_tensor(
        torch.arange(dim, device=device).repeat(2, 1),
        torch.ones(dim, dtype=Cplx, device=device),
        (dim, dim),
    )


class BaseGate(nn.Module):
    def __init__(self, dim=2, index=[0], dits=1, inverse=False, device="cpu"):
        super().__init__()
        self.index = index
        self.device = device
        self.inverse = inverse
        self.dits = dits
        self.dims = [dim] * dits if isinstance(dim, int) else dim
        self.total_dim = int(np.prod(self.dims))

    def _infer_dits(self, x):
        if isinstance(self.dims[0], int) and len(set(self.dims)) == 1:
            return int(round(log(x.shape[0], self.dims[0])))
        return len(self.dims)

    def _getUnitary(self, x, gate_matrices):
        L = self._infer_dits(x)
        U = eye(1, device=x.device)
        for i in range(L):
            if i in self.index:
                M = (
                    gate_matrices[i]
                    if isinstance(gate_matrices, dict)
                    else gate_matrices
                )
            else:
                M = eye(self.dims[i], device=x.device)
            U = Ut.kron(U, M)
        return U


class SingleDitGate(BaseGate):
    def __init__(self, dim=2, index=[0], dits=1, inverse=False, device="cpu", **kwargs):
        super().__init__(dim, index, dits, inverse, device)
        self.M_dict = self._getMat(**kwargs)

    def _getMat(self, **kwargs):
        raise NotImplementedError

    def forward(self, x):
        U = self._getUnitary(x, self.M_dict)
        return U @ x

    def matrix(self):
        L = self.dits
        U = eye(1, device=self.device)
        for i in range(L):
            if i in self.index:
                M = self.M_dict[i] if isinstance(self.M_dict, dict) else self.M_dict
            else:
                M = eye(self.dims[i], device=self.device)
            U = Ut.kron(U, M)
        return U


class H(SingleDitGate):
    def _getMat(self, **kwargs):
        M_dict = {}
        for idx in self.index:
            d = self.dims[idx]
            omega = np.exp(2 * 1j * np.pi / d)
            M = torch.ones((d, d), dtype=Cplx, device=self.device)
            for i in range(1, d):
                for j in range(1, d):
                    M[i, j] = omega ** (i * j)
            M = M / (d**0.5)
            if self.inverse:
                M = torch.conj(M).T.contiguous()
            M_dict[idx] = M.to_sparse()
        return M_dict


class X(SingleDitGate):
    def _getMat(self, s=1, **kwargs):
        M_dict = {}
        for idx in self.index:
            d = self.dims[idx]
            M = torch.zeros((d, d), dtype=Cplx, device=self.device)
            for i in range(d):
                j = (i + s) % d
                M[j, i] = 1.0
            if self.inverse:
                M = torch.conj(M.T)
            M_dict[idx] = M.to_sparse()
        return M_dict


class Z(SingleDitGate):
    def _getMat(self, s=1, **kwargs):
        M_dict = {}
        for idx in self.index:
            d = self.dims[idx]
            omega = np.exp(2 * 1j * np.pi / d)
            M = torch.diag(
                torch.tensor(
                    [omega ** (j * s) for j in range(d)],
                    dtype=Cplx,
                    device=self.device,
                )
            )
            if self.inverse:
                M = torch.conj(M.T)
            M_dict[idx] = M.to_sparse()
        return M_dict


class Y(SingleDitGate):
    def _getMat(self, s=1, **kwargs):
        M_dict = {}
        x_gate = X(
            dim=self.dims[0] if len(set(self.dims)) == 1 else self.dims,
            index=self.index,
            dits=self.dits,
            device=self.device,
            s=s,
        )
        z_gate = Z(
            dim=self.dims[0] if len(set(self.dims)) == 1 else self.dims,
            index=self.index,
            dits=self.dits,
            device=self.device,
            s=s,
        )
        for i in self.index:
            M = torch.matmul(z_gate.M_dict[i], x_gate.M_dict[i]) / 1j
            M_dict[i] = M.to_sparse()
        return M_dict


class ParametrizedRotation(BaseGate):
    def __init__(self, j=0, k=1, index=[0], dim=2, dits=1, device="cpu", angle=None):
        super().__init__(dim, index, dits, False, device)
        assert angle is not None, "angle parameter is required and cannot be None"
        self.j_map = self._build_level_map(j)
        self.k_map = self._build_level_map(k) if k is not None else None
        if isinstance(angle, torch.Tensor):
            self.angle = angle
        else:
            self.angle = torch.tensor(angle, device=device, dtype=torch.float32)
        if self.angle.numel() == 1:
            self.angle = self.angle.expand(self.dits)
        elif self.angle.numel() != self.dits:
            raise ValueError(f"Angle must have {self.dits} elements or be a scalar")

    def _build_level_map(self, levels):
        if isinstance(levels, int):
            return {t: levels for t in self.index}
        if len(levels) != len(self.index):
            raise ValueError("Level list length must equal number of target dits")
        return {t: level for t, level in zip(self.index, levels)}

    def forward(self, x, param=None):
        L = self._infer_dits(x)
        U = eye(1, device=x.device)
        for i in range(L):
            d = self.dims[i]
            if i in self.index:
                j_val = self.j_map[i]
                k_val = self.k_map[i] if self.k_map else None
                angle_val = (
                    self.angle[i]
                    if param is None
                    else torch.tensor(param[i], device=self.device)
                )
                M = self._getRMat(d, j_val, k_val, angle_val)
            else:
                M = eye(d, device=x.device)
            U = Ut.kron(U, M)
        return U @ x


class RX(ParametrizedRotation):
    def _getRMat(self, d, j_val, k_val, angle_val):
        indices = torch.tensor(
            [[j_val, k_val, j_val, k_val], [j_val, k_val, k_val, j_val]],
            device=self.device,
        )
        values = torch.tensor(
            [
                torch.cos(angle_val / 2),
                torch.cos(angle_val / 2),
                -1j * torch.sin(angle_val / 2),
                -1j * torch.sin(angle_val / 2),
            ],
            dtype=Cplx,
            device=self.device,
        )
        M = eye(d, device=self.device)
        return Ut.sparse_index_put(M, indices, values, self.device)


class RY(ParametrizedRotation):
    def _getRMat(self, d, j_val, k_val, angle_val):
        indices = torch.tensor(
            [[j_val, k_val, j_val, k_val], [j_val, k_val, k_val, j_val]],
            device=self.device,
        )
        values = torch.tensor(
            [
                torch.cos(angle_val / 2),
                torch.cos(angle_val / 2),
                -torch.sin(angle_val / 2),
                torch.sin(angle_val / 2),
            ],
            dtype=Cplx,
            device=self.device,
        )
        M = eye(d, device=self.device)
        return Ut.sparse_index_put(M, indices, values, self.device)


class RZ(ParametrizedRotation):
    def __init__(self, j=1, index=[0], dim=2, dits=1, device="cpu", angle=None):
        super().__init__(j, None, index, dim, dits, device, angle)

    def _getRMat(self, d, j_val, k_val, angle_val):
        if d == 2:
            phases = [torch.exp(1j * angle_val / 2), torch.exp(-1j * angle_val / 2)]
            if j_val == 1:
                phases = phases[::-1]
            M = torch.diag(torch.tensor(phases, dtype=Cplx, device=self.device))
            return M.to_sparse()
        else:
            M = eye(d, device=self.device)
            phase = torch.exp(1j * angle_val)
            indices = torch.tensor([[j_val], [j_val]], device=self.device)
            values = torch.tensor([phase], dtype=Cplx, device=self.device)
            return Ut.sparse_index_put(M, indices, values, self.device)


class TwoQuditGate(BaseGate):
    def __init__(self, index=[0, 1], dits=2, dim=2, device="cpu", inverse=False):
        super().__init__(dim, index, dits, inverse, device)
        self.U = self._makeMat()

    def forward(self, x):
        return self.U @ x

    def matrix(self):
        return self.U


class CX(TwoQuditGate):
    def _makeMat(self):
        U = Ut.CX_sparse(
            self.index[0],
            self.index[1],
            self.dims[0] if len(set(self.dims)) == 1 else self.dims,
            self.dits,
            device=self.device,
        )
        if self.inverse:
            U = torch.conj(U).T.contiguous()
        return U


class CZ(TwoQuditGate):
    def _makeMat(self):
        D = self.total_dim
        U = torch.zeros((D, D), device=self.device, dtype=Cplx)
        control_dim = self.dims[self.index[0]]
        for c_val in range(control_dim):
            u = torch.eye(1, device=self.device, dtype=Cplx)
            for i in range(self.dits):
                if i == self.index[0]:
                    dim = self.dims[i]
                    proj = torch.eye(dim, device=self.device, dtype=Cplx).reshape(
                        (dim, dim, 1)
                    )
                    P = proj @ proj.T.conj()
                    u = torch.kron(u, P)
                elif i == self.index[1]:
                    M = Z(dim=self.dims[i], device=self.device, s=c_val).matrix()
                    u = torch.kron(u, M)
                else:
                    u = torch.kron(
                        u,
                        torch.eye(self.dims[i], device=self.device, dtype=Cplx),
                    )
            U += u
        return U.to_sparse()


class SWAP(TwoQuditGate):
    def _makeMat(self):
        c, t = self.index[0], self.index[1]
        D = self.total_dim
        U = torch.zeros((D, D), device=self.device, dtype=Cplx)
        for k in range(D):
            localr = Ut.dec2den(k, self.dits, self.dims)
            locall = localr.copy()
            locall[c], locall[t] = localr[t], localr[c]
            globall = Ut.den2dec(locall, self.dims)
            U[globall, k] = 1
        return U.to_sparse()


class MultiControlGate(BaseGate):
    def __init__(self, index=[0, 1, 2], dim=2, dits=3, inverse=False, device="cpu"):
        super().__init__(dim, index, dits, inverse, device)
        self.U = self._build_multi_control_matrix()

    def forward(self, x):
        return self.U @ x

    def matrix(self):
        return self.U


class CCX(MultiControlGate):
    def _build_multi_control_matrix(self):
        D = self.total_dim
        basis = torch.tensor(list(product(*[range(d) for d in self.dims]))).to(
            self.device
        )
        basis_modified = basis.clone()
        target_dim = self.dims[self.index[2]]
        basis_modified[:, self.index[2]] = (
            basis[:, self.index[0]] * basis[:, self.index[1]] + basis[:, self.index[2]]
        ) % target_dim
        eq_matrix = torch.all(basis[:, None, :] == basis_modified[None, :, :], dim=2)
        U = torch.where(
            eq_matrix,
            torch.tensor(1.0 + 0j, dtype=Cplx, device=self.device),
            torch.tensor(0.0, dtype=Cplx, device=self.device),
        )
        if self.inverse:
            U = torch.conj(U).T.contiguous()
        return U.to_sparse()


class MCX(MultiControlGate):
    def _build_multi_control_matrix(self):
        basis = torch.tensor(list(product(*[range(d) for d in self.dims]))).to(
            self.device
        )
        basis_modified = basis.clone()
        control_value = 1
        for i in range(len(self.index) - 1):
            control_value *= basis_modified[:, self.index[i]]
        target_dim = self.dims[self.index[-1]]
        basis_modified[:, self.index[-1]] = (
            control_value + basis_modified[:, self.index[-1]]
        ) % target_dim
        eq_matrix = torch.all(basis[:, None, :] == basis_modified[None, :, :], dim=2)
        U = torch.where(
            eq_matrix,
            torch.tensor(1.0 + 0j, dtype=Cplx, device=self.device),
            torch.tensor(0.0, dtype=Cplx, device=self.device),
        )
        if self.inverse:
            U = torch.conj(U).T.contiguous()
        return U.to_sparse()


class U(BaseGate):
    def __init__(self, matrix, dim=2, dits=1, device="cpu", index=None):
        assert matrix is not None, "matrix parameter is required"
        index = index or list(range(dits))
        super().__init__(dim, index, dits, False, device)

        self.indices = sorted(index)
        self.dims_ = [dim] * dits  # enforce uniform dims
        self.sub_dim = dim ** len(self.indices)
        self.total_dim = dim**dits

        # Precompute permutation metadata
        self.perm = self.indices + [i for i in range(dits) if i not in self.indices]
        self.inv_perm = np.argsort(self.perm)
        self.shape_perm = [self.dims_[i] for i in self.perm]

        # Matrix validation and conversion
        self.M = (
            matrix
            if isinstance(matrix, torch.Tensor)
            else torch.tensor(matrix, device=device, dtype=Cplx)
        )
        if self.M.shape != (self.sub_dim, self.sub_dim):
            raise ValueError(
                f"Expected shape {(self.sub_dim, self.sub_dim)}, got {self.M.shape}"
            )
        self.M = self.M.to(device)

    def forward(self, x):
        x = x.view(*self.dims_)  # assume correct shape
        x = x.permute(*self.perm).reshape(self.sub_dim, -1)
        x = self.M @ x
        x = x.view(*self.shape_perm).permute(*self.inv_perm).reshape(self.total_dim, 1)
        return x


class CU(BaseGate):
    def __init__(
        self, dim=2, dits=2, device="cpu", index=[0, 1], matrix=None, control_dim=None
    ):
        if index is None or len(index) < 2:
            raise ValueError(
                "Index must have at least two elements: [control, target(s)]"
            )
        assert matrix is not None, "matrix parameter is required and cannot be None"
        super().__init__(dim, index, dits, False, device)
        self.control_index = index[0]
        self.target_index = index[1:]
        self.d_control = self.dims[self.control_index]
        self.d_target = int(np.prod([self.dims[i] for i in self.target_index]))
        self.sub_dim = self.d_control * self.d_target
        if control_dim is None:
            self.control_dim = [1] if self.d_control == 2 else [self.d_control - 1]
        elif isinstance(control_dim, int):
            self.control_dim = [control_dim]
        else:
            self.control_dim = list(control_dim)
        for ctrl_state in self.control_dim:
            if ctrl_state < 0 or ctrl_state >= self.d_control:
                raise ValueError(
                    f"Control state {ctrl_state} out of range [0, {self.d_control-1}]"
                )
        self.custom_blocks = self._process_matrix_input(matrix)

    def _process_matrix_input(self, matrix):
        custom_blocks = {}
        if isinstance(matrix, list):
            if len(matrix) != len(self.control_dim):
                raise ValueError("Matrix list length must equal control_dim length")
            for i, ctrl_state in enumerate(self.control_dim):
                if isinstance(matrix[i], torch.Tensor):
                    M = matrix[i]
                else:
                    M = torch.tensor(matrix[i], device=self.device, dtype=Cplx)
                if M.shape != (self.d_target, self.d_target):
                    raise ValueError(
                        f"Matrix {i} shape must be ({self.d_target}, {self.d_target})"
                    )
                custom_blocks[ctrl_state] = M
        elif isinstance(matrix, torch.Tensor):
            if matrix.ndim == 3:
                if matrix.shape[0] != len(self.control_dim):
                    raise ValueError(
                        "3D tensor first dimension must equal control_dim length"
                    )
                for i, ctrl_state in enumerate(self.control_dim):
                    block = matrix[i]
                    if block.shape != (self.d_target, self.d_target):
                        raise ValueError(
                            f"Block {i} shape must be ({self.d_target}, {self.d_target})"
                        )
                    custom_blocks[ctrl_state] = block
            elif matrix.ndim == 2:
                expected = len(self.control_dim) * self.d_target
                if matrix.shape != (expected, expected):
                    raise ValueError(
                        f"2D tensor shape must be ({expected}, {expected})"
                    )
                reshaped = matrix.view(
                    len(self.control_dim), self.d_target, self.d_target
                )
                for i, ctrl_state in enumerate(self.control_dim):
                    custom_blocks[ctrl_state] = reshaped[i]
            else:
                raise ValueError("Matrix tensor must be 2D or 3D")
        else:
            raise ValueError("Matrix must be list or torch.Tensor")
        return custom_blocks

    def _get_control_blocks(self):
        blocks = []
        for k in range(self.d_control):
            if k in self.control_dim:
                U_k = self.custom_blocks[k]
            else:
                U_k = torch.eye(self.d_target, dtype=Cplx, device=self.device)
            blocks.append(U_k)
        return blocks

    def forward(self, x):
        remaining = [i for i in range(self.dits) if i not in self.index]
        new_order = self.index + remaining
        inv_order = [new_order.index(i) for i in range(self.dits)]
        state_tensor = x.view(*self.dims)
        psi_perm = state_tensor.permute(*new_order).contiguous()
        d_sub = int(np.prod([self.dims[i] for i in self.index]))
        d_rem = int(np.prod([self.dims[i] for i in remaining])) if remaining else 1
        A = psi_perm.view(d_sub, d_rem)
        A = A.view(self.d_control, self.d_target, d_rem)
        blocks = self._get_control_blocks()
        U_sub = torch.block_diag(*blocks)
        A_sub = A.view(self.sub_dim, d_rem)
        A_new = U_sub @ A_sub
        new_shape = (
            [self.d_control]
            + [self.dims[i] for i in self.target_index]
            + ([self.dims[i] for i in remaining] if remaining else [])
        )
        psi_perm_new = A_new.view(*new_shape)
        psi_final = psi_perm_new.permute(*inv_order).contiguous().view(-1, 1)

        return psi_final


def I(matrix, dim=2, index=[0], dits=1, device="cpu"):
    assert matrix is not None, "matrix parameter is required for I gate"
    return U(matrix=matrix, dim=dim, dits=dits, device=device, index=index)


def S(matrix, dim=2, index=[0], dits=1, device="cpu"):
    assert matrix is not None, "matrix parameter is required for S gate"
    return U(matrix=matrix, dim=dim, dits=dits, device=device, index=index)


def T(matrix, dim=2, index=[0], dits=1, device="cpu"):
    assert matrix is not None, "matrix parameter is required for T gate"
    return U(matrix=matrix, dim=dim, dits=dits, device=device, index=index)


def P(matrix, dim=2, index=[0], dits=1, device="cpu"):
    assert matrix is not None, "matrix parameter is required for P gate"
    return U(matrix=matrix, dim=dim, dits=dits, device=device, index=index)


class Gategen:
    def __init__(self, dim=2, device="cpu"):
        self.dim = dim
        self.device = device

    @property
    def I(self):
        return torch.eye(self.dim, dtype=Cplx, device=self.device)

    @property
    def H(self):
        h_gate = H(dim=self.dim, index=[0], dits=1, device=self.device)
        return h_gate.M_dict[0]

    @property
    def X(self):
        x_gate = X(dim=self.dim, index=[0], dits=1, device=self.device)
        return x_gate.M_dict[0]

    @property
    def Z(self):
        z_gate = Z(dim=self.dim, index=[0], dits=1, device=self.device)
        return z_gate.M_dict[0]

    @property
    def Y(self):
        y_gate = Y(dim=self.dim, index=[0], dits=1, device=self.device)
        return y_gate.M_dict[0]

    @property
    def S(self):
        omega = np.exp(2 * 1j * np.pi / (self.dim * 2))
        return torch.diag(
            torch.tensor(
                [omega**j for j in range(self.dim)],
                dtype=Cplx,
                device=self.device,
            )
        )

    @property
    def T(self):
        omega = np.exp(2 * 1j * np.pi / (self.dim * 4))
        return torch.diag(
            torch.tensor(
                [omega**j for j in range(self.dim)],
                dtype=Cplx,
                device=self.device,
            )
        )

    def P(self, theta):
        phases = [np.exp(1j * theta * j / (self.dim - 1)) for j in range(self.dim)]
        return torch.diag(torch.tensor(phases, dtype=Cplx, device=self.device))

    def RX(self, j, k, angle):
        rx_gate = RX(
            j=j, k=k, index=[0], dim=self.dim, dits=1, device=self.device, angle=angle
        )
        return rx_gate._getRMat(self.dim, j, k, angle)

    def RY(self, j, k, angle):
        ry_gate = RY(
            j=j, k=k, index=[0], dim=self.dim, dits=1, device=self.device, angle=angle
        )
        return ry_gate._getRMat(self.dim, j, k, angle)

    def RZ(self, j, angle):
        rz_gate = RZ(
            j=j, index=[0], dim=self.dim, dits=1, device=self.device, angle=angle
        )
        return rz_gate._getRMat(self.dim, j, None, angle)

    @property
    def CX(self):
        return CX

    @property
    def CZ(self):
        return CZ

    @property
    def SWAP(self):
        return SWAP

    @property
    def CCX(self):
        return CCX

    def make(self, matrix, name=None):
        def gate_func(dim=2, dits=1, index=None, **kwargs):
            if index is None:
                if isinstance(dim, int):
                    matrix_shape = np.array(matrix).shape[0]
                    num_qudits = int(np.log(matrix_shape) / np.log(dim))
                    if dim**num_qudits != matrix_shape:
                        raise ValueError(
                            f"Matrix dimension {matrix_shape} doesn't match dim^n for any integer n"
                        )
                    index = list(range(num_qudits))
                else:
                    index = list(range(len(dim)))

            if isinstance(matrix, torch.Tensor):
                pmatrix = matrix
            elif isinstance(matrix, np.ndarray):
                pmatrix = torch.from_numpy(matrix).to(self.device, dtype=Cplx)
            else:
                pmatrix = torch.tensor(matrix, device=self.device, dtype=Cplx)

            pmatrix = pmatrix.to_sparse()
            return U(
                matrix=pmatrix,
                dim=dim,
                dits=dits,
                device=self.device,
                index=index,
            )

        if name:
            gate_func.__name__ = name
        return gate_func
