from itertools import product
import torch.nn as nn
from math import log
import numpy as np
import torch

Cplx = torch.complex64


def tensorise(m, device="cpu", dtype=Cplx):
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


def CX_sparse(ctrl_index, targ_idx, dim, wires, device):
    if isinstance(dim, int) and wires > 1:
        dim = [dim] * wires

    total_dim = int(np.prod(dim))
    U = torch.eye(total_dim, dtype=Cplx, device=device)
    dimlist = [dim] * wires if isinstance(dim, int) else dim
    for i in range(total_dim):
        state_n = dec2den(i, wires, dimlist)

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
    def __init__(self, dim=2, index=[0], wires=1, inverse=False, device="cpu"):
        super().__init__()
        self.index = index
        self.device = device
        self.inverse = inverse
        self.wires = wires
        self.dims = [dim] * wires if isinstance(dim, int) else dim
        self.total_dim = int(np.prod(self.dims))
        self.omega = (
            torch.tensor(np.exp(2 * 1j * np.pi / dim), dtype=Cplx, device=device)
            if isinstance(dim, int)
            else None
        )

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
    def __init__(
        self, dim=2, index=[0], wires=1, inverse=False, device="cpu", **kwargs
    ):
        super().__init__(dim, index, wires, inverse, device)
        self.M_dict = self._getMat(**kwargs)

    def _getMat(self, **kwargs):
        raise NotImplementedError

    def forward(self, x):
        U = self._getUnitary(x, self.M_dict)
        return U @ x

    def matrix(self):
        L = self.wires
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
            omega = torch.tensor(
                np.exp(2 * 1j * np.pi / d), dtype=Cplx, device=self.device
            )
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
            omega = torch.tensor(
                np.exp(2 * 1j * np.pi / d), dtype=Cplx, device=self.device
            )

            def fill_fn(dim):
                return torch.diag(
                    torch.tensor(
                        [omega ** (j * s) for j in range(dim)],
                        dtype=Cplx,
                        device=self.device,
                    )
                )

            M_dict[idx] = self._make_matrix(d, fill_fn)
        return M_dict


class Y(SingleDitGate):
    def _getMat(self, s=1, **kwargs):
        M_dict = {}
        x_gate = X(
            dim=self.dims[0] if len(set(self.dims)) == 1 else self.dims,
            index=self.index,
            wires=self.wires,
            device=self.device,
            s=s,
        )
        z_gate = Z(
            dim=self.dims[0] if len(set(self.dims)) == 1 else self.dims,
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
        super().__init__(dim, index, wires, False, device)
        assert angle is not None, "angle parameter is required and cannot be None"

        self.j_map = self._build_level_map(j)
        self.k_map = self._build_level_map(k) if k is not None else None

        if isinstance(angle, torch.Tensor):
            self.angle = angle
        else:
            self.angle = torch.tensor(angle, device=device, dtype=torch.float32)
        if self.angle.numel() == 1:
            self.angle = self.angle.expand(self.wires)
        elif self.angle.numel() != self.wires:
            raise ValueError(
                f"Expected {self.wires} angles, got {self.angle.numel()}", self.angle
            )

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
    def __init__(self, index=[0], dim=2, wires=1, device="cpu", angle=None):
        super().__init__(0, 1, index, dim, wires, device, angle)

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
    def __init__(self, index=[0], dim=2, wires=1, device="cpu", angle=None):
        super().__init__(0, 1, index, dim, wires, device, angle)

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
    def __init__(self, j=1, index=[0], dim=2, wires=1, device="cpu", angle=None):
        super().__init__(j, None, index, dim, wires, device, angle)

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
    def __init__(self, j, k, index=[0], dim=2, wires=1, device="cpu", angle=None):
        super().__init__(
            j=j, k=k, index=index, dim=dim, wires=wires, device=device, angle=angle
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


def T(dim=2, index=[0], wires=1, device="cpu"):
    omega = torch.tensor(np.exp(2 * 1j * np.pi / (dim * 4)), dtype=Cplx, device=device)
    matrix = torch.diag(
        torch.tensor([omega**j for j in range(dim)], dtype=Cplx, device=device)
    )

    return U(matrix=matrix, dim=dim, wires=wires, device=device, index=index)


def S(dim=2, index=[0], wires=1, device="cpu"):
    omega = torch.tensor(np.exp(2 * 1j * np.pi / (dim * 2)), dtype=Cplx, device=device)
    matrix = torch.diag(
        torch.tensor([omega**j for j in range(dim)], dtype=Cplx, device=device)
    )

    return U(matrix=matrix, dim=dim, wires=wires, device=device, index=index)


def P(theta, dim=2, index=[0], wires=1, device="cpu"):
    phases = [np.exp(1j * theta * j / (dim - 1)) for j in range(dim)]
    matrix = torch.diag(torch.tensor(phases, dtype=Cplx, device=device))

    return U(matrix=matrix, dim=dim, wires=wires, device=device, index=index)


class CX(BaseGate):
    def __init__(
        self, index=[0, 1], wires=2, dim=2, device="cpu", sparse=False, inverse=False
    ):
        super().__init__(dim, index, wires, inverse, device)
        if len(self.index) != 2:
            raise ValueError("CNOT gate requires exactly two index: [control, target].")
        self.control_idx = self.index[0]
        self.target_idx = self.index[1]
        self.sparse = sparse
        self.U = self._build_cnot_matrix()
        self.register_buffer("U_matrix", self._apply_inverse(self.U))

    def _build_cnot_matrix(self):
        L = torch.tensor(list(product(*[range(d) for d in self.dims]))).to(
            self.device, non_blocking=True
        )
        l2ns = L.clone()
        d_target = self.dims[self.target_idx]

        l2ns[:, self.target_idx] = (
            L[:, self.control_idx] + L[:, self.target_idx]
        ) % d_target

        index_mask = torch.all(L[:, None, :] == l2ns[None, :, :], dim=2)
        U = torch.where(
            index_mask,
            torch.tensor(1.0 + 0j, dtype=Cplx, device=self.device),
            torch.tensor(0.0, dtype=Cplx, device=self.device),
        )
        return U

    def forward(self, x):
        return self.U_matrix @ x

    def matrix(self):
        return self.U_matrix


class CZ(BaseGate):
    def __init__(self, index=[0, 1], dim=2, wires=2, device="cpu"):
        super().__init__(dim, index, wires, False, device)
        if len(self.index) != 2:
            raise ValueError("CZ gate requires exactly two index: [control, target].")
        self.control_idx = self.index[0]
        self.target_idx = self.index[1]
        self.U = self._build_cz_matrix()
        self.register_buffer("U_matrix", self.U)

    def _build_cz_matrix(self):
        U = torch.zeros(
            (self.total_dim, self.total_dim), device=self.device, dtype=Cplx
        )
        control_dim = self.dims[self.control_idx]

        for c_val in range(control_dim):
            u_block = self._eye(1)
            for i in range(self.wires):
                if i == self.control_idx:
                    proj_vec = torch.eye(
                        self.dims[i], dtype=Cplx, device=self.device
                    ).unsqueeze(1)[c_val]

                    P = proj_vec @ proj_vec.T.conj()
                    u_block = torch.kron(u_block, P)
                elif i == self.target_idx:
                    M = Z(dim=self.dims[i], device=self.device, s=c_val).matrix()
                    u_block = torch.kron(u_block, M)
                else:
                    u_block = torch.kron(u_block, self._eye(self.dims[i]))
            U += u_block
        return U

    def forward(self, x):
        return self.U_matrix @ x

    def matrix(self):
        return self.U_matrix


class SWAP(BaseGate):
    """
    SWAP gate for qudits.
    Exchanges the states of two qudits.
    """

    def __init__(self, index=[0, 1], dim=2, wires=2, device="cpu"):
        super().__init__(dim, index, wires, False, device)
        if len(self.index) != 2:
            raise ValueError("SWAP gate requires exactly two index: [qudit1, qudit2].")
        self.q1_idx = self.index[0]
        self.q2_idx = self.index[1]
        self.U = self._build_swap_matrix()
        self.register_buffer("U_matrix", self.U)

    def _build_swap_matrix(self):
        U = torch.zeros(
            (self.total_dim, self.total_dim), device=self.device, dtype=Cplx
        )
        for k in range(self.total_dim):
            localr = dec2den(k, self.wires, self.dims)
            locall = list(localr)

            locall[self.q1_idx], locall[self.q2_idx] = (
                localr[self.q2_idx],
                localr[self.q1_idx],
            )

            globall = den2dec(locall, self.dims)
            U[globall, k] = 1
        return U

    def forward(self, x):
        return self.U_matrix @ x

    def matrix(self):
        return self.U_matrix


class CCX(BaseGate):
    """
    Controlled-Controlled-NOT (CCNOT) or Toffoli gate for qudits.
    Applies an X gate to the target qudit if both control qudits are in state '1'
    (or the last state for d > 2).
    """

    def __init__(self, index=[0, 1, 2], dim=2, wires=3, inverse=False, device="cpu"):
        super().__init__(dim, index, wires, inverse, device)
        if len(self.index) != 3:
            raise ValueError(
                "CCNOT gate requires exactly three index: [control1, control2, target]."
            )
        self.control1_idx = self.index[0]
        self.control2_idx = self.index[1]
        self.target_idx = self.index[2]
        self.U = self._build_ccnot_matrix()
        self.register_buffer("U_matrix", self._apply_inverse(self.U))

    def _build_ccnot_matrix(self):
        basis = torch.tensor(list(product(*[range(d) for d in self.dims]))).to(
            self.device, non_blocking=True
        )
        basis_modified = basis.clone()
        target_dim = self.dims[self.target_idx]

        basis_modified[:, self.target_idx] = (
            basis[:, self.control1_idx] * basis[:, self.control2_idx]
            + basis[:, self.target_idx]
        ) % target_dim

        eq_matrix = torch.all(basis[:, None, :] == basis_modified[None, :, :], dim=2)
        U = torch.where(
            eq_matrix,
            torch.tensor(1.0 + 0j, dtype=Cplx, device=self.device),
            torch.tensor(0.0, dtype=Cplx, device=self.device),
        )
        return U

    def forward(self, x):
        return self.U_matrix @ x

    def matrix(self):
        return self.U_matrix


class U(BaseGate):
    """
    Arbitrary unitary gate. Can be applied to a subset of qudits or the entire system.
    Can be initialized with a fixed matrix or a trainable random unitary.
    """

    def __init__(self, matrix=None, dim=2, wires=1, device="cpu", index=None):
        super().__init__(
            dim,
            index if index is not None else list(range(wires)),
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

        self.random = matrix is None

        if self.index is None:
            self.sub_dims = self.dims
            self.sub_dim = self.total_dim
        else:
            self.sub_dims = [self.dims[i] for i in self.index]
            self.sub_dim = int(np.prod(self.sub_dims))

        if self.random:
            initial_matrix = torch.eye(
                self.sub_dim, device=device, dtype=Cplx
            ) + torch.randn((self.sub_dim, self.sub_dim), device=device, dtype=Cplx)
            self.U_param = nn.Parameter(initial_matrix)
        else:
            self.M = tensorise(matrix, device=device, dtype=Cplx)
            if self.M.shape != (self.sub_dim, self.sub_dim):
                raise ValueError(
                    "Provided matrix dimensions do not match the product of the targeted qudits' dimensions."
                )

    def _get_unitary_from_param(self):
        """Constructs a unitary matrix from the trainable parameter."""
        skew_hermitian_part = self.U_param - torch.conj(self.U_param.T)
        return torch.matrix_exp(skew_hermitian_part)

    def forward(self, x):
        if self.index is None:
            U_final = self._get_unitary_from_param() if self.random else self.M
            return U_final @ x
        else:
            all_index = list(range(self.wires))
            target_index = self.index
            remaining_index = [i for i in all_index if i not in target_index]
            new_order = target_index + remaining_index
            inv_order = [new_order.index(i) for i in range(self.wires)]

            psi = x.view(*self.dims)
            psi_perm = psi.permute(*new_order).contiguous()

            d_rest = self.total_dim // self.sub_dim
            psi_flat = psi_perm.reshape(self.sub_dim, d_rest)

            U_sub = self._get_unitary_from_param() if self.random else self.M
            psi_transformed = U_sub @ psi_flat

            new_shape = [self.dims[i] for i in new_order]
            psi_perm_transformed = psi_transformed.reshape(*new_shape)
            psi_final = psi_perm_transformed.permute(*inv_order).contiguous()
            return psi_final.view(self.total_dim, 1)

    def matrix(self):
        if self.index is None:
            return self._get_unitary_from_param() if self.random else self.M
        else:
            U_sub = self._get_unitary_from_param() if self.random else self.M

            basis_index = list(product(*[range(d) for d in self.dims]))
            perm = []
            all_index = list(range(self.wires))
            target_index = self.index
            remaining_index = [i for i in all_index if i not in target_index]
            new_order = target_index + remaining_index
            new_dims = [self.dims[i] for i in new_order]

            for m in basis_index:
                m_list = list(m)
                permuted = [m_list[i] for i in new_order]
                new_dec = den2dec(permuted, new_dims)
                perm.append(new_dec)

            perm = torch.tensor(perm, dtype=torch.long, device=self.device)
            P = torch.zeros(
                (self.total_dim, self.total_dim), dtype=Cplx, device=self.device
            )
            for i in range(self.total_dim):
                P[i, perm[i]] = 1.0

            d_rest = self.total_dim // self.sub_dim
            I_rest = torch.eye(d_rest, dtype=Cplx, device=self.device)
            U_embedded = torch.kron(U_sub, I_rest)

            U_full = P.T @ U_embedded @ P
            return U_full


class CU(BaseGate):
    """
    Controlled-U (CU) gate for qudits.
    Applies a unitary U to the target qudit(s) if the control qudit is in a specific state(s).
    """

    def __init__(
        self, dim=2, wires=2, device="cpu", index=[0, 1], matrix=None, control_dim=None
    ):
        super().__init__(dim, index, wires, False, device)
        if index is None or len(index) < 2:
            raise ValueError(
                "The 'index' parameter must be a list with at least two elements: one control and at least one target."
            )

        self.control_index = index[0]
        self.target_index = index[1:]

        self.d_control = self.dims[self.control_index]
        self.d_target = int(np.prod([self.dims[i] for i in self.target_index]))
        self.sub_dim_local = self.d_control * self.d_target

        if control_dim is None:
            self.control_states = [self.d_control - 1]
        elif isinstance(control_dim, int):
            self.control_states = [control_dim]
        else:
            self.control_states = list(control_dim)

        for ctrl_state in self.control_states:
            if ctrl_state < 0 or ctrl_state >= self.d_control:
                raise ValueError(
                    "Each element in control_dim must lie in range [0, d_control-1]."
                )

        self.random = matrix is None
        if self.random:
            num_active_blocks = len(self.control_states)
            self.U_blocks_param = nn.Parameter(
                torch.randn(
                    num_active_blocks,
                    self.d_target,
                    self.d_target,
                    device=device,
                    dtype=Cplx,
                )
            )
        else:
            self.custom_blocks = {}
            if isinstance(matrix, list):
                if len(matrix) != len(self.control_states):
                    raise ValueError(
                        "When providing a list, the number of matrices must equal len(control_dim)."
                    )
                for i, ctrl_state in enumerate(self.control_states):
                    M = torch.tensor(matrix[i], device=device, dtype=Cplx)
                    if M.shape != (self.d_target, self.d_target):
                        raise ValueError(
                            "Each provided matrix must be of shape (d_target, d_target)."
                        )
                    self.custom_blocks[ctrl_state] = M
            elif isinstance(matrix, torch.Tensor):
                if matrix.ndim == 3:
                    if matrix.shape[0] != len(self.control_states):
                        raise ValueError(
                            "For a 3D tensor, the first dimension must equal len(control_dim)."
                        )
                    for i, ctrl_state in enumerate(self.control_states):
                        block = matrix[i]
                        if block.shape != (self.d_target, self.d_target):
                            raise ValueError(
                                "Each provided block must have shape (d_target, d_target)."
                            )
                        self.custom_blocks[ctrl_state] = block
                elif matrix.ndim == 2:
                    expected_dim = len(self.control_states) * self.d_target
                    if (
                        matrix.shape[0] != expected_dim
                        or matrix.shape[1] != expected_dim
                    ):
                        raise ValueError(
                            "For a 2D tensor, the shape must be (len(control_dim)*d_target, len(control_dim)*d_target)."
                        )
                    reshaped = matrix.view(
                        len(self.control_states), self.d_target, self.d_target
                    )
                    for i, ctrl_state in enumerate(self.control_states):
                        self.custom_blocks[ctrl_state] = reshaped[i]
                else:
                    raise ValueError(
                        "Provided matrix must be either a list of matrices, a 3D tensor, or a 2D tensor."
                    )
            else:
                raise ValueError(
                    "Provided matrix must be either a list or a torch.Tensor."
                )

    def _get_target_unitary(self, k, idx_in_control_states):
        """Returns the unitary for the target qudit(s) based on the control state k."""
        if self.random:
            param = self.U_blocks_param[idx_in_control_states]
            return torch.matrix_exp(param - torch.conj(param).T)
        else:
            return self.custom_blocks[k]

    def forward(self, x):
        all_index = list(range(self.wires))
        sub_index = [self.control_index] + self.target_index
        remaining_index = [i for i in all_index if i not in sub_index]
        new_order = sub_index + remaining_index
        inv_order = [new_order.index(i) for i in range(self.wires)]

        state_tensor = x.view(*self.dims)
        psi_perm = state_tensor.permute(*new_order).contiguous()

        d_sub_permuted_section = int(np.prod([self.dims[i] for i in sub_index]))
        d_rem = (
            int(np.prod([self.dims[i] for i in remaining_index]))
            if remaining_index
            else 1
        )
        A = psi_perm.view(d_sub_permuted_section, d_rem)
        A = A.view(self.d_control, self.d_target, d_rem)

        blocks_to_apply = []
        for k in range(self.d_control):
            if k in self.control_states:
                idx = self.control_states.index(k)
                U_k = self._get_target_unitary(k, idx)
            else:
                U_k = self._eye(self.d_target)
            blocks_to_apply.append(U_k)

        U_sub_combined = torch.block_diag(*blocks_to_apply)

        A_sub = A.view(self.sub_dim_local, d_rem)
        A_new = U_sub_combined @ A_sub

        new_perm_shape = [self.dims[i] for i in new_order]
        psi_perm_new = A_new.view(*new_perm_shape)

        psi_final = psi_perm_new.permute(*inv_order).contiguous().view(-1, 1)
        return psi_final

    def matrix(self):
        blocks_for_matrix = []
        for k in range(self.d_control):
            if k in self.control_states:
                idx = self.control_states.index(k)
                U_k = self._get_target_unitary(k, idx)
            else:
                U_k = self._eye(self.d_target)
            blocks_for_matrix.append(U_k)
        U_sub_combined = torch.block_diag(*blocks_for_matrix)

        all_index = list(range(self.wires))
        sub_index = [self.control_index] + self.target_index
        remaining_index = [i for i in all_index if i not in sub_index]
        d_rem = (
            int(np.prod([self.dims[i] for i in remaining_index]))
            if remaining_index
            else 1
        )
        I_rem = torch.eye(d_rem, dtype=Cplx, device=self.device)
        U_embedded = torch.kron(U_sub_combined, I_rem)

        new_order = sub_index + remaining_index
        new_dims = [self.dims[i] for i in new_order]

        basis_index = list(product(*[range(d) for d in self.dims]))
        perm = []
        for m in basis_index:
            m_list = list(m)
            permuted = [m_list[i] for i in new_order]
            new_dec = den2dec(permuted, new_dims)
            perm.append(new_dec)

        perm = torch.tensor(perm, dtype=torch.long, device=self.device)
        P = torch.zeros(
            (self.total_dim, self.total_dim), dtype=Cplx, device=self.device
        )
        for i in range(self.total_dim):
            P[i, perm[i]] = 1.0

        U_full = P.T @ U_embedded @ P
        return U_full


class Gategen:
    def __init__(self, dim=2, device="cpu"):
        self.dim = dim
        self.device = device

    @property
    def I(self):
        return torch.eye(self.dim, dtype=Cplx, device=self.device)

    @property
    def H(self):
        h_gate = H(dim=self.dim, index=[0], wires=1, device=self.device)
        return h_gate.M_dict[0]

    @property
    def X(self):
        x_gate = X(dim=self.dim, index=[0], wires=1, device=self.device)
        return x_gate.M_dict[0]

    @property
    def Z(self):
        z_gate = Z(dim=self.dim, index=[0], wires=1, device=self.device)
        return z_gate.M_dict[0]

    @property
    def Y(self):
        y_gate = Y(dim=self.dim, index=[0], wires=1, device=self.device)
        return y_gate.M_dict[0]

    @property
    def S(self):
        omega = torch.tensor(
            np.exp(2 * 1j * np.pi / (self.dim * 2)), dtype=Cplx, device=self.device
        )
        return torch.diag(
            torch.tensor(
                [omega**j for j in range(self.dim)], dtype=Cplx, device=self.device
            )
        )

    @property
    def T(self):
        omega = torch.tensor(
            np.exp(2 * 1j * np.pi / (self.dim * 4)), dtype=Cplx, device=self.device
        )
        return torch.diag(
            torch.tensor(
                [omega**j for j in range(self.dim)], dtype=Cplx, device=self.device
            )
        )

    def P(self, theta):
        phases = [np.exp(1j * theta * j / (self.dim - 1)) for j in range(self.dim)]
        return torch.diag(torch.tensor(phases, dtype=Cplx, device=self.device))

    def RX(self, j, k, angle):
        rx_gate = RX(index=[0], dim=self.dim, wires=1, device=self.device, angle=angle)
        return rx_gate._getRMat(self.dim, j, k, angle)

    def RY(self, j, k, angle):
        ry_gate = RY(index=[0], dim=self.dim, wires=1, device=self.device, angle=angle)
        return ry_gate._getRMat(self.dim, j, k, angle)

    def RZ(self, j, angle):
        rz_gate = RZ(
            j=j, index=[0], dim=self.dim, wires=1, device=self.device, angle=angle
        )
        return rz_gate._getRMat(self.dim, j, None, angle)

    @property
    def CX(self):
        cx_gate = CX(dim=self.dim, index=[0, 1], wires=2, device=self.device)
        cx_gate = cx_gate.matrix()
        cx_gate.__name__ = "CX"
        return cx_gate

    @property
    def CZ(self):
        cz_gate = CZ(dim=self.dim, index=[0, 1], wires=2, device=self.device)
        return cz_gate.matrix()

    @property
    def SWAP(self):
        swap_gate = SWAP(dim=self.dim, index=[0, 1], wires=2, device=self.device)
        return swap_gate.matrix()

    @property
    def CCX(self):
        ccx_gate = CCX(dim=self.dim, index=[0, 1, 2], wires=3, device=self.device)
        return ccx_gate.matrix()

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
                pmatrix = torch.tensor(matrix, device=self.device, dtype=Cplx)

            return U(
                matrix=pmatrix,
                dim=dim,
                wires=wires,
                device=self.device,
                index=index,
            )

        gate_func.__name__ = "GG.make"
        return gate_func
