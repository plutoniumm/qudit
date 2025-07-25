from itertools import product
import torch.nn as nn
from math import log
import numpy as np
import torch

Cplx = torch.complex64

def dec2den(dec, dits, dims):
    den = []
    temp_dec = dec
    for i in range(dits - 1, -1, -1):
        den.insert(0, temp_dec % dims[i])
        temp_dec //= dims[i]
    return den


def den2dec(den, dims):
    dec = 0
    for i in range(len(den)):
        stride = int(np.prod(dims[i + 1 :])) if i < len(den) - 1 else 1
        dec += den[i] * stride
    return dec


def CX_sparse(ctrl_index, targ_idx, dim, dits, device):
    if isinstance(dim, int) and dits > 1:
        dim = [dim] * dits

    total_dim = int(np.prod(dim))
    U = torch.eye(total_dim, dtype=Cplx, device=device)
    dimlist = [dim] * dits if isinstance(dim, int) else dim
    for i in range(total_dim):
        state_n = dec2den(i, dits, dimlist)

        if state_n[ctrl_index] == dimlist[ctrl_index] - 1:
            target = state_n[targ_idx]
            new_target = (target + 1) % dimlist[targ_idx]
            new_state_n = list(state_n)
            new_state_n[targ_idx] = new_target
            j = den2dec(new_state_n, dimlist)
            U[j, i] = 1.0 + 0j
            U[i, i] = 0.0 + 0j
    return U


class BaseGate(nn.Module):
    def __init__(self, dim=2, index=[0], dits=1, inverse=False, device="cpu"):
        super().__init__()
        self.index = index
        self.device = device
        self.inverse = inverse
        self.dits = dits
        self.dims = [dim] * dits if isinstance(dim, int) else dim
        self.total_dim = int(np.prod(self.dims))
        self.omega = torch.tensor(np.exp(2 * 1j * np.pi / dim), dtype=Cplx, device=device) if isinstance(dim, int) else None

    def _eye(self, dim):
        return torch.eye(dim, dtype=Cplx, device=self.device)

    def _make_matrix(self, dim, fill_fn):
        M = fill_fn(dim)
        return self._apply_inverse(M)

    def _apply_inverse(self, M):
        return torch.conj(M).T.contiguous() if self.inverse else M

    def _infer_dits(self, x):
        if isinstance(self.dims[0], int) and len(set(self.dims)) == 1:
            return int(round(log(x.shape[0], self.dims[0])))
        return len(self.dims)

    def _getUnitary(self, x, gate_matrices):
        L = self._infer_dits(x)
        U = self._eye(1)
        for i in range(L):
            if i in self.index:
                M = (
                    gate_matrices[i]
                    if isinstance(gate_matrices, dict)
                    else gate_matrices
                )
            else:
                M = self._eye(self.dims[i])
            U = torch.kron(U, M)
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
        U = self._eye(1)
        for i in range(L):
            if i in self.index:
                M = self.M_dict[i] if isinstance(self.M_dict, dict) else self.M_dict
            else:
                M = self._eye(self.dims[i])
            U = torch.kron(U, M)
        return U


class H(SingleDitGate):
    def _getMat(self, **kwargs):
        M_dict = {}
        for idx in self.index:
            d = self.dims[idx]
            omega = torch.tensor(np.exp(2 * 1j * np.pi / d), dtype=Cplx, device=self.device)
            M = torch.ones((d, d), dtype=Cplx, device=self.device)
            for i in range(1, d):
                for j in range(1, d):
                    M[i, j] = omega ** (i * j)
            M = M / (d**0.5)
            M_dict[idx] = self._apply_inverse(M)
        return M_dict


class X(SingleDitGate):
    def _getMat(self, s=1, **kwargs):
        M_dict = {}
        for idx in self.index:
            d = self.dims[idx]
            def fill_fn(dim):
                M = torch.zeros((dim, dim), dtype=Cplx, device=self.device)
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
            omega = torch.tensor(np.exp(2 * 1j * np.pi / d), dtype=Cplx, device=self.device)
            def fill_fn(dim):
                return torch.diag(torch.tensor([omega ** (j * s) for j in range(dim)], dtype=Cplx, device=self.device))
            M_dict[idx] = self._make_matrix(d, fill_fn)
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
            M_dict[i] = M
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
        U_total = self._eye(1)
        for i in range(L):
            d = self.dims[i]
            if i in self.index:
                j_val = self.j_map[i]
                k_val = self.k_map[i] if self.k_map else None
                angle_val = self.angle[i] if param is None else param[i]
                M_block = self._getRMat(d, j_val, k_val, angle_val)
            else:
                M_block = self._eye(d)
            U_total = torch.kron(U_total, M_block)
        return U_total @ x


class RX(ParametrizedRotation):
    def __init__(self, index=[0], dim=2, dits=1, device="cpu", angle=None):
        super().__init__(0, 1, index, dim, dits, device, angle)

    def _getRMat(self, d, j_val, k_val, angle_val):
        M = self._eye(d)
        cos_hf = torch.cos(angle_val / 2)
        sin_hf_i = -1j * torch.sin(angle_val / 2)
        M[j_val, j_val] = cos_hf
        M[k_val, k_val] = cos_hf
        M[j_val, k_val] = sin_hf_i
        M[k_val, j_val] = sin_hf_i
        return M


class RY(ParametrizedRotation):
    def __init__(self, index=[0], dim=2, dits=1, device="cpu", angle=None):
        super().__init__(0, 1, index, dim, dits, device, angle)

    def _getRMat(self, d, j_val, k_val, angle_val):
        M = self._eye(d)
        cos_hf = torch.cos(angle_val / 2)
        sin_hf = torch.sin(angle_val / 2)
        M[j_val, j_val] = cos_hf
        M[k_val, k_val] = cos_hf
        M[j_val, k_val] = -sin_hf
        M[k_val, j_val] = sin_hf
        return M


class RZ(ParametrizedRotation):
    def __init__(self, j=1, index=[0], dim=2, dits=1, device="cpu", angle=None):
        super().__init__(j, None, index, dim, dits, device, angle)

    def _getRMat(self, d, j_val, k_val, angle_val):
        if d == 2:
            phases = torch.empty(d, dtype=Cplx, device=self.device)
            phases[0] = torch.exp(1j * angle_val / 2)
            phases[1] = torch.exp(-1j * angle_val / 2)
            if j_val == 1:
                phases = phases.flip(dims=[0])
            M = torch.diag(phases)
            return M
        else:
            M = self._eye(d)
            phase = torch.exp(1j * angle_val)
            M[j_val, j_val] = phase
            return M


class GellMann:
    def __init__(self, j, k, d):
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
    def __init__(self, j, k, index=[0], dim=2, dits=1, device="cpu", angle=None):
        super().__init__(
            j=j, k=k, index=index, dim=dim, dits=dits, device=device, angle=angle
        )

    def _getRMat(self, d, j_val, k_val, angle_val):
        gm = GellMann(j_val + 1, k_val + 1, d).matrix
        gm_tensor = torch.tensor(gm, dtype=Cplx, device=self.device)

        M = self._eye(d)
        c, s = torch.cos(angle_val / 2), torch.sin(angle_val / 2)
        M[j_val, j_val] = c
        M[k_val, k_val] = c
        M[j_val, k_val] = -1j * s * gm_tensor[j_val, k_val]
        M[k_val, j_val] = -1j * s * gm_tensor[k_val, j_val]
        return M


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
        U = CX_sparse(
            self.index[0],
            self.index[1],
            self.dims[0] if len(set(self.dims)) == 1 else self.dims,
            self.dits,
            device=self.device,
        )
        return self._apply_inverse(U)


class CZ(TwoQuditGate):
    def _makeMat(self):
        D = self.total_dim
        U = torch.zeros((D, D), device=self.device, dtype=Cplx)
        ctrl_dim = self.dims[self.index[0]]
        for c_val in range(ctrl_dim):
            u = torch.eye(1, device=self.device, dtype=Cplx)
            for i in range(self.dits):
                if i == self.index[0]:
                    dim = self.dims[i]
                    P = torch.zeros((dim, dim), dtype=Cplx, device=self.device)
                    P[c_val, c_val] = 1.0
                    u = torch.kron(u, P)
                elif i == self.index[1]:
                    M = Z(dim=self.dims[i], device=self.device, s=c_val).matrix()
                    u = torch.kron(u, M)
                else:
                    u = torch.kron(u, self._eye(self.dims[i]))
            U += u
        return U


class SWAP(TwoQuditGate):
    def _makeMat(self):
        c, t = self.index[0], self.index[1]
        D = self.total_dim
        U = torch.zeros((D, D), device=self.device, dtype=Cplx)
        for k in range(D):
            localr = dec2den(k, self.dits, self.dims)
            locall = localr.copy()
            locall[c], locall[t] = localr[t], localr[c]
            globall = den2dec(locall, self.dims)
            U[globall, k] = 1
        return U


class MultictrlGate(BaseGate):
    def __init__(self, index=[0, 1, 2], dim=2, dits=3, inverse=False, device="cpu"):
        super().__init__(dim, index, dits, inverse, device)
        self.U = self._build_multi_ctrl_matrix()

    def forward(self, x):
        return self.U @ x

    def matrix(self):
        return self.U


class CCX(MultictrlGate):
    def _build_multi_ctrl_matrix(self):
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
        return self._apply_inverse(U)


class MCX(MultictrlGate):
    def _build_multi_ctrl_matrix(self):
        basis = torch.tensor(list(product(*[range(d) for d in self.dims]))).to(
            self.device
        )
        basis_modified = basis.clone()
        ctrl_value = 1
        for i in range(len(self.index) - 1):
            ctrl_value *= basis_modified[:, self.index[i]]
        target_dim = self.dims[self.index[-1]]
        basis_modified[:, self.index[-1]] = (
            ctrl_value + basis_modified[:, self.index[-1]]
        ) % target_dim
        eq_matrix = torch.all(basis[:, None, :] == basis_modified[None, :, :], dim=2)
        U = torch.where(
            eq_matrix,
            torch.tensor(1.0 + 0j, dtype=Cplx, device=self.device),
            torch.tensor(0.0, dtype=Cplx, device=self.device),
        )
        return self._apply_inverse(U)


class U(BaseGate):
    def __init__(self, matrix, dim=2, dits=1, device="cpu", index=None):
        assert matrix is not None, "matrix parameter is required"
        index = index or list(range(dits))
        super().__init__(dim, index, dits, False, device)
        self.indices = sorted(index)
        if isinstance(dim, int):
            self.dims_ = [dim] * dits
        elif isinstance(dim, list) and len(dim) == dits:
            self.dims_ = dim
        else:
            raise ValueError(
                "dim must be an integer or a list with length equal to dits"
            )
        self.sub_dim = int(np.prod([self.dims_[i] for i in self.indices]))
        self.total_dim = int(np.prod(self.dims_))
        self.perm = self.indices + [i for i in range(dits) if i not in self.indices]
        self.inv_perm = np.argsort(self.perm)
        self.shape_perm = [self.dims_[i] for i in self.perm]
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
        x = x.view(*self.dims_)
        x = x.permute(*self.perm).contiguous().reshape(self.sub_dim, -1)
        x = self.M @ x
        x = (
            x.view(*self.shape_perm)
            .permute(*self.inv_perm)
            .contiguous()
            .reshape(self.total_dim, 1)
        )
        return x


class CU(BaseGate):
    def __init__(
        self, dim=2, dits=2, device="cpu", index=[0, 1], matrix=None, ctrl_dim=None
    ):
        if index is None or len(index) < 2:
            raise ValueError(
                "Index must have at least two elements: [ctrl, target(s)]"
            )
        assert matrix is not None, "matrix parameter is required and cannot be None"
        super().__init__(dim, index, dits, False, device)
        self.ctrl_index = index[0]
        self.targ_idx = index[1:]
        self.d_ctrl = self.dims[self.ctrl_index]
        self.d_target = int(np.prod([self.dims[i] for i in self.targ_idx]))
        self.sub_dim = self.d_ctrl * self.d_target

        if ctrl_dim is None:
            self.ctrl_dim = [1] if self.d_ctrl == 2 else [self.d_ctrl - 1]
        elif isinstance(ctrl_dim, int):
            self.ctrl_dim = [ctrl_dim]
        else:
            self.ctrl_dim = list(ctrl_dim)

        for ctrl_state in self.ctrl_dim:
            if ctrl_state < 0 or ctrl_state >= self.d_ctrl:
                raise ValueError(
                    f"ctrl state {ctrl_state} out of range [0, {self.d_ctrl-1}]"
                )

        self.custom_blocks = self._process_matrix_input(matrix)

    def _process_matrix_input(self, matrix):
        custom_blocks = {}
        if isinstance(matrix, list):
            if len(matrix) != len(self.ctrl_dim):
                raise ValueError("Matrix list length must equal ctrl_dim length")
            for i, ctrl_state in enumerate(self.ctrl_dim):
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
                if matrix.shape[0] != len(self.ctrl_dim):
                    raise ValueError(
                        "3D tensor first dimension must equal ctrl_dim length"
                    )
                for i, ctrl_state in enumerate(self.ctrl_dim):
                    block = matrix[i]
                    if block.shape != (self.d_target, self.d_target):
                        raise ValueError(
                            f"Block {i} shape must be ({self.d_target}, {self.d_target})"
                        )
                    custom_blocks[ctrl_state] = block
            elif matrix.ndim == 2:
                expected = len(self.ctrl_dim) * self.d_target
                if matrix.shape != (expected, expected):
                    raise ValueError(
                        f"2D tensor shape must be ({expected}, {expected})"
                    )
                reshaped = matrix.view(
                    len(self.ctrl_dim), self.d_target, self.d_target
                )
                for i, ctrl_state in enumerate(self.ctrl_dim):
                    custom_blocks[ctrl_state] = reshaped[i]
            else:
                raise ValueError("Matrix tensor must be 2D or 3D")
        else:
            raise ValueError("Matrix must be list or torch.Tensor")
        return custom_blocks

    def _get_ctrl_blocks(self):
        blocks = []
        for k in range(self.d_ctrl):
            if k in self.ctrl_dim:
                U_k = self.custom_blocks[k]
            else:
                U_k = self._eye(self.d_target)
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
        A = psi_perm.view(self.d_ctrl, self.d_target, d_rem)
        blocks = self._get_ctrl_blocks()
        U_sub = torch.block_diag(*blocks)
        A_sub = A.view(self.sub_dim, d_rem)
        A_new = U_sub @ A_sub
        new_shape = (
            [self.d_ctrl]
            + [self.dims[i] for i in self.targ_idx]
            + ([self.dims[i] for i in remaining] if remaining else [])
        )
        psi_perm_new = A_new.view(*new_shape)
        psi_final = psi_perm_new.permute(*inv_order).contiguous().view(-1, 1)
        return psi_final


# Specialized gate functions that provide meaningful quantum operations
def T(dim=2, index=[0], dits=1, device="cpu"):
    """T gate - pi/8 phase gate (fourth root of Z)"""
    gate = BaseGate(dim, index, dits, device=device)
    # T gate is Z^(1/4), so omega^(1/4)
    omega = torch.tensor(np.exp(2 * 1j * np.pi / (dim * 4)), dtype=Cplx, device=device)
    matrix = torch.diag(torch.tensor([omega**j for j in range(dim)], dtype=Cplx, device=device))
    return U(matrix=matrix, dim=dim, dits=dits, device=device, index=index)


def S(dim=2, index=[0], dits=1, device="cpu"):
    """S gate - pi/4 phase gate (square root of Z)"""
    gate = BaseGate(dim, index, dits, device=device)
    # S gate is Z^(1/2), so omega^(1/2)
    omega = torch.tensor(np.exp(2 * 1j * np.pi / (dim * 2)), dtype=Cplx, device=device)
    matrix = torch.diag(torch.tensor([omega**j for j in range(dim)], dtype=Cplx, device=device))
    return U(matrix=matrix, dim=dim, dits=dits, device=device, index=index)


def P(theta, dim=2, index=[0], dits=1, device="cpu"):
    """P gate - parameterized phase gate"""
    phases = [np.exp(1j * theta * j / (dim - 1)) for j in range(dim)]
    matrix = torch.diag(torch.tensor(phases, dtype=Cplx, device=device))
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
        omega = torch.tensor(np.exp(2 * 1j * np.pi / (self.dim * 2)), dtype=Cplx, device=self.device)
        return torch.diag(torch.tensor([omega**j for j in range(self.dim)], dtype=Cplx, device=self.device))

    @property
    def T(self):
        omega = torch.tensor(np.exp(2 * 1j * np.pi / (self.dim * 4)), dtype=Cplx, device=self.device)
        return torch.diag(torch.tensor([omega**j for j in range(self.dim)], dtype=Cplx, device=self.device))

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
        name = "U" if not hasattr(matrix, "name") else matrix.name

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
            pmatrix = (
                matrix
                if isinstance(matrix, torch.Tensor)
                else torch.tensor(matrix, device=self.device, dtype=Cplx)
            )
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