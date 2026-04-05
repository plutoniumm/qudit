from functools import cached_property
from typing import List, Optional, Union, Callable, Any, Tuple
from .. import State
import torch.nn as nn
import numpy as np
import torch
import cmath
import math

C64 = torch.complex64


class Operator:
    """
    Generic gate class wrapping a complex matrix/tensor with a name and optional parameters/metadata.

    - tensor product via '^'
    - matrix composition via '@'.

    """

    tensor: torch.Tensor
    name: str
    params: List[Any]

    def __init__(
        self, tensor: torch.Tensor, name: str, params: Optional[List[Any]] = None
    ):
        self.tensor = tensor
        self.name = name
        self.params = list(params) if params is not None else []

    def __getattr__(self, item: str) -> Any:
        return getattr(self.tensor, item)

    def __repr__(self) -> str:
        """
        Human Readable representation with name and optionally parameters.
        """
        if self.params:
            return f"{self.name!r}({self.params})"
        else:
            return self.name

    def __xor__(self, other: Any) -> Any:
        # Operator ^ Operator
        if isinstance(other, Operator):
            tensor = torch.kron(self.tensor, other.tensor)
            name = f"{self.name} ^ {other.name}"
            params = self.params + other.params

            return Operator(tensor, name, params)

        # Operator ^ Gate
        if isinstance(other, Gate):
            tensor = torch.kron(self.tensor, other.matrix())
            name = f"{self.name} ^ {other.name}"
            params = self.params + list(other.params)

            return Operator(tensor, name, params)

        # Operator ^ Tensor
        if isinstance(other, torch.Tensor):
            tensor = torch.kron(self.tensor, other)
            name = f"{self.name} ^ Tensor"

            return Operator(tensor, name)

        # Operator ^ State
        if isinstance(other, State):
            tensor = torch.kron(
                self.tensor, other.tensor if hasattr(other, "tensor") else other
            )

            return State(tensor)

        raise TypeError("Can only tensor product with Operator, Gate, Tensor, or State")

    def __rxor__(self, other: Any) -> Union["Operator", Any]:
        if isinstance(other, Gate):
            tensor = torch.kron(other.matrix(), self.tensor)
            name = f"{other.name} ^ {self.name}"
            params = list(other.params) + list(self.params)

            return Operator(tensor, name, params)

        return NotImplemented

    def __matmul__(self, other: Any) -> Any:
        # Operator @ State
        if isinstance(other, State):
            return State(
                self.tensor @ (other.tensor if hasattr(other, "tensor") else other)
            )

        # Operator @ Operator
        elif isinstance(other, Operator):
            tensor = self.tensor @ other.tensor
            name = f"{self.name} @ {other.name}"
            params = self.params + other.params

            return Operator(tensor, name, params)

        # Operator @ Tensor
        elif isinstance(other, torch.Tensor):
            tensor = self.tensor @ other
            name = f"{self.name} @ Tensor"

            return Operator(tensor, name)
        else:
            raise TypeError("Can only apply to State or compose with another Operator")

    def __rmatmul__(self, other: Any) -> Any:
        matmul = getattr(other, "__matmul__", None)

        if callable(matmul):
            return matmul(self)

        raise TypeError(
            f"Operator '@' not supported between {type(other)} and Operator"
        )


def tensorise(m: Any, device: str = "cpu", dtype: torch.dtype = C64) -> torch.Tensor:
    """
    Convert common array-likes (Tensor/ndarray/list/Operator) into a torch complex tensor.
    """
    if isinstance(m, torch.Tensor):
        return m.to(device=device, dtype=dtype)
    elif isinstance(m, np.ndarray):
        return torch.from_numpy(m).to(device, non_blocking=True).type(dtype)
    elif isinstance(m, list):
        return torch.tensor(m, device=device, dtype=dtype)
    elif isinstance(m, Operator):
        return m.tensor.to(device=device, dtype=dtype)
    else:
        raise TypeError(
            f"Unsupported type: {type(m)}. Expected Tensor, ndarray, or list."
        )


def gell_mann(j: int, k: int, d: int, device: str = "cpu") -> torch.Tensor:
    """
    Return a generalized Gell-Mann generator $\lambda_{jk}$ for $SU(d)$.
    """
    m = torch.zeros((d, d), dtype=C64, device=device)

    if j < k:
        m[j, k] = 1.0
        m[k, j] = 1.0
    elif j > k:
        m[k, j] = -1j
        m[j, k] = 1j
    else:
        l = j + 1
        if l >= d:
            return torch.eye(d, dtype=C64, device=device)

        scale = math.sqrt(2 / (l * (l + 1)))
        for i in range(l):
            m[i, i] = scale
        m[l, l] = -l * scale

    return m


class Gate(nn.Module):
    """
    Parameterized unitary acting on a subset of wires, embedded into the full Hilbert space. A generic container class which handles all the reshaping/permuting logic to apply a target-space unitary to the correct subset of wires in a larger system.

    The full dense matrix can be materialized if needed, but the main use is to apply the unitary to states or compose with other gates without ever explicitly constructing the full matrix. The target-space matrix is stored as a parameter and can be optimized over if desired.
    """

    device: str
    wires: int
    index: List[int]
    dims: List[int]
    name: str
    params: List[Any]
    total_dim: int
    target_dims: List[int]
    target_size: int
    U: torch.Tensor
    all: List[int]
    unused: List[int]
    perm: List[int]
    inv_perm: List[int]
    rest_size: int

    def __init__(
        self,
        matrix: Any,
        index: List[int],
        wires: int,
        dim: Union[int, List[int]],
        device: str = "cpu",
        name: str = "U",
        params: Optional[list] = None,
    ):
        super().__init__()
        self.device = device
        self.wires = wires
        self.index = index if isinstance(index, list) else [index]
        self.dims = [dim] * wires if isinstance(dim, int) else dim
        self.name = name
        self.params = list(params) if params is not None else []

        self.total_dim = math.prod(self.dims)
        self.target_dims = [self.dims[i] for i in self.index]
        self.target_size = math.prod(self.target_dims)

        self.U = tensorise(matrix, device=device)
        if self.U.shape != (self.target_size, self.target_size):
            raise ValueError(
                f"Matrix shape {self.U.shape} does not match target size {(self.target_size, self.target_size)}."
            )

        self.all = list(range(self.wires))
        self.unused = [i for i in self.all if i not in self.index]
        self.perm = self.index + self.unused

        self.inv_perm = list(np.argsort(self.perm))

        self.rest_size = self.total_dim // self.target_size

    def _left(
        self, tensor: torch.Tensor, U: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generalized core logic: applies the embedded unitary U to the first dimension of a tensor.
        Handles both 1D statevectors (D,) and 2D matrices (D, 1) or (D, D).
        Pass an explicit U to override self.U (used by NoisyGate for live Kraus operators).
        """
        if U is None:
            U = self.U

        # Determine any trailing dimensions (e.g., the second D in a density matrix)
        trailing_dims = list(tensor.shape[1:])

        # Reshape to (*self.dims, *trailing_dims).
        # Using contiguous() prevents view errors if the tensor was previously transposed.
        psi = tensor.contiguous().view(*self.dims, *trailing_dims)

        # Permute target wires to the front, keep trailing dimensions at the end
        trailing_axes = list(range(self.wires, self.wires + len(trailing_dims)))
        perm = self.perm + trailing_axes
        psi = psi.permute(*perm)

        # Flatten for matrix multiplication: (target_size, rest_size * prod(trailing_dims))
        trailing_size = math.prod(trailing_dims) if trailing_dims else 1
        psi_flat = psi.reshape(self.target_size, self.rest_size * trailing_size)

        # Apply the target unitary
        psi_out = U @ psi_flat

        # Reshape back to the permuted structure
        current_dims = [self.dims[i] for i in self.perm]
        psi_out = psi_out.view(*current_dims, *trailing_dims)

        # Inverse permute to original wire order
        inv_perm = self.inv_perm + trailing_axes
        psi_final = psi_out.permute(*inv_perm).contiguous()

        return psi_final.view(self.total_dim, *trailing_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the embedded unitary to a statevector: $|\psi\\rangle \mapsto U|\psi\\rangle$.
        """
        out = self._left(x)
        # Ensure it outputs a column vector (D, 1) to match your original API behavior

        return out.view(self.total_dim, 1)

    def forwardd(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Apply the unitary channel to a density matrix: $\\rho \mapsto U\\rho U^\dagger$.
        """
        # 1. Apply U on the left: U \rho
        rho_prime = self._left(rho)

        # 2. Conjugate transpose the result: (U \rho)^\dagger
        rho_prime_dagger = rho_prime.conj().T

        # 3. Apply U on the left again: U (U \rho)^\dagger
        out_dagger = self._left(rho_prime_dagger)

        # 4. Conjugate transpose back to get: (U (U \rho)^\dagger)^\dagger = U \rho U^\dagger

        return out_dagger.conj().T

    def matrix(self) -> torch.Tensor:
        """
        Materialize the full dense matrix by applying the module to computational basis vectors.
        """
        eye = torch.eye(self.total_dim, device=self.device, dtype=C64)
        cols = []
        for i in range(self.total_dim):
            cols.append(self.forward(eye[i]))

        return torch.cat(cols, dim=1)

    def __matmul__(self, other: Any) -> Any:
        # State
        if isinstance(other, State):
            return State(
                self.forward(other.tensor if hasattr(other, "tensor") else other)
            )

        # Operator
        elif isinstance(other, Operator):
            U_full = self.matrix()
            mat = U_full @ tensorise(other.tensor, device=self.device)
            name = f"{self.name} @ {other.name}"
            params = list(self.params) + list(other.params)

            return Operator(mat, name, params)

        # Tensor-like
        elif isinstance(other, torch.Tensor):
            U_full = self.matrix()
            mat = U_full @ other.to(device=self.device, dtype=C64)
            name = f"{self.name} @ Tensor"

            return Operator(mat, name)
        else:
            raise TypeError("Can only apply Gate to State, Operator, or Tensor")


class Gategen:
    """
    Convenience factory for common qudit gates at a fixed local dimension.
    """

    dim: int
    device: str

    def __init__(self, dim: int = 2, device: str = "cpu"):
        self.dim = dim
        self.device = device

    @cached_property
    def I(self) -> Operator:
        """
        Identity gate on one qudit such that $I|k\\rangle = |k\\rangle$ for all states $|k\\rangle$.
        """
        return Operator(torch.eye(self.dim, dtype=C64, device=self.device), "I")

    @cached_property
    def H(self) -> Operator:
        """
        Hadamard/DFT gate (H for d=2, discrete Fourier transform for $d$>2) such that $H|k\\rangle = \\frac{1}{\sqrt{d}} \sum_{j=0}^{d-1} \omega^{jk} |j\\rangle$ where $\omega = e^{2\\pi i / d}$.
        """
        d = self.dim
        if d == 2:
            m = torch.tensor(
                [[1, 1], [1, -1]], dtype=C64, device=self.device
            ) / math.sqrt(2)
        else:
            w = cmath.exp(2j * math.pi / d)
            idx = torch.arange(d, device=self.device)
            m = (w ** torch.outer(idx, idx)) / math.sqrt(d)

        return Operator(m.to(dtype=C64), "H")

    @cached_property
    def X(self) -> Operator:
        """
        Generalized X (cyclic shift) gate as $X|k\\rangle = |k+1 \mod d\\rangle$ (Pauli-X for $d=2$).
        """
        if self.dim == 2:
            m = torch.tensor([[0, 1], [1, 0]], dtype=C64, device=self.device)
        else:
            m = torch.roll(
                torch.eye(self.dim, dtype=C64, device=self.device), shifts=-1, dims=1
            )

        return Operator(m, "X")

    @cached_property
    def Z(self) -> Operator:
        """
        Generalized Z (phase) gate: $Z|k\\rangle = \omega^k|k\\rangle$ where $\omega = e^{2\pi i/d}$ (Pauli-$Z$ for $d=2$).
        """
        d = self.dim
        if d == 2:
            m = torch.tensor([[1, 0], [0, -1]], dtype=C64, device=self.device)
        else:
            w = cmath.exp(2j * math.pi / d)
            idx = torch.arange(d, device=self.device)
            m = torch.diag(w**idx)

        return Operator(m, "Z")

    @cached_property
    def Y(self) -> Operator:
        """
        Generalized Y (up to phase), built from Z and X (Pauli-Y for d=2) as $Y = Z X / i$ such that $Y|k\\rangle = -i \omega^k |k+1 \mod d\\rangle$.
        """
        d = self.dim
        if d == 2:
            m = torch.tensor([[0, -1j], [1j, 0]], dtype=C64, device=self.device)
        else:
            m = torch.matmul(self.Z.tensor, self.X.tensor) / 1j

        return Operator(m, "Y")

    def inCircuit(self, kwargs: dict) -> bool:
        """
        Return True if kwargs contain circuit-embedding keys (index/wires/dim).
        """
        return all(k in kwargs for k in ("index", "wires", "dim"))

    def GMR(
        self,
        j: int,
        k: int,
        angle: Any,
        type: str = "asym",
        *,
        matrix: bool = False,
        **kwargs: Any,
    ) -> Union[Operator, Gate]:
        """
        Generalized rotation from a Gell-Mann generator (symmetric/asymmetric/diagonal). Gell-Mann gates are described by their type (sym/asym/diag) and the indices j, k specifying the generator.

        We generate each of them as $GMR_{\\text{sym}}(j,k,\\theta) = \\exp(-i \\frac{\\theta}{2} \\lambda_{jk}^{\\text{sym}})$, $GMR_{\\text{asym}}(j,k,\\theta) = \\exp(-i \\frac{\\theta}{2} \\lambda_{jk}^{\\text{asym}})$, and $GMR_{\\text{diag}}(j,j,\\theta) = \\exp(-i \\frac{\\theta}{2} \\lambda_{jj}^{\\text{diag}})$ where $\\lambda_{jk}$ are the Gell-Mann generators.
        """
        if not self.inCircuit(kwargs):
            matrix = True

        orig_angle = angle
        if not isinstance(angle, torch.Tensor):
            angle = torch.tensor(angle, dtype=C64, device=self.device)
        else:
            angle = angle.to(device=self.device)

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
            c = torch.cos(angle / 2).to(dtype=C64)
            s = torch.sin(angle / 2).to(dtype=C64)

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
            ang = angle.to(dtype=C64)
            m = torch.matrix_exp(-1j * (ang / 2) * gen)
            gate_name = f"GMR_{type}"

        gate_params: List[Tuple[str, Any]] = [
            ("type", type),
            ("j", j),
            ("k", k),
            ("angle", angle),
            ("dim", self.dim),
        ]

        if matrix:
            return Operator(m, gate_name, params=gate_params)

        if not self.inCircuit(kwargs):
            raise TypeError(
                f"{gate_name} missing circuit kwargs (index/wires/dim). "
                f"Call with matrix=True for standalone Operator."
            )

        index = kwargs.pop("index")
        wires = kwargs.pop("wires")
        dim = kwargs.pop("dim")
        name = kwargs.pop("name", None)

        needs_grad = isinstance(orig_angle, nn.Parameter) or (
            isinstance(orig_angle, torch.Tensor) and orig_angle.requires_grad
        )

        if needs_grad:
            build_fn = _gmr_factory(j, k, type, self.dim, self.device)

            return VarGate(
                build_fn,
                orig_angle,
                index=index,
                wires=wires,
                dim=dim,
                device=self.device,
                name=name or gate_name,
                params=gate_params,
            )

        return Gate(
            m,
            index=index,
            wires=wires,
            dim=dim,
            device=self.device,
            name=name or gate_name,
            params=gate_params,
        )

    def RX(
        self, angle: Any, *, matrix: bool = False, **kwargs: Any
    ) -> Union[Operator, Gate]:
        """
        Rotation in the (0,1) symmetric subspace (qubit-like Rx when d=2) as $RX(\\theta) = GMR_{\\text{sym}}(0,1,\\theta)$.
        """
        return self.GMR(0, 1, angle, type="sym", matrix=matrix, **kwargs)

    def RY(
        self, angle: Any, *, matrix: bool = False, **kwargs: Any
    ) -> Union[Operator, Gate]:
        """
        Rotation in the (0,1) asymmetric subspace (qubit-like Ry when d=2) as $RY(\\theta) = GMR_{\\text{asym}}(0,1,\\theta)$.
        """
        return self.GMR(0, 1, angle, type="asym", matrix=matrix, **kwargs)

    def RZ(
        self, angle: Any, *, matrix: bool = False, **kwargs: Any
    ) -> Union[Operator, Gate]:
        """
        Diagonal generator rotation (qubit-like Rz when d=2) as $RZ(\\theta) = GMR_{\\text{diag}}(0,0,\\theta)$.
        """
        return self.GMR(0, 0, angle, type="diag", matrix=matrix, **kwargs)

    def CU(
        self, U_target: Any = None, *, matrix: bool = False, **kwargs: Any
    ) -> Union[Operator, Gate]:
        """
        Controlled-unitary: apply target block when control is in a chosen computational state such that when $U_\mathrm{target}$ is a $d \\times d$ unitary, $CU = |0\\rangle\langle 0| \otimes I + |1\\rangle\langle 1| \otimes U$ (generalized CNOT for $U=X$ and $d=2$).
        """
        d = self.dim

        U_mat = tensorise(
            U_target.tensor if isinstance(U_target, Operator) else U_target,
            device=self.device,
        )
        if U_mat.shape != (d, d):
            raise ValueError(f"U_target must be a ({d},{d}) matrix, got {U_mat.shape}.")

        I = torch.eye(d, device=self.device, dtype=C64)
        blocks = [I]
        for k in range(1, d):
            blocks.append(blocks[-1] @ U_mat)

        m = torch.block_diag(*blocks)
        gate_name = "CU"

        target_name = U_target.name if isinstance(U_target, Operator) else None
        gate_params: List[Tuple[str, Any]] = [
            ("target", target_name),
            ("dim", d),
        ]

        if matrix:
            return Operator(m, gate_name, params=gate_params)

        index = kwargs.pop("index")
        wires = kwargs.pop("wires")
        dim = kwargs.pop("dim")
        name = kwargs.pop("name", None)

        return Gate(
            m,
            index=index,
            wires=wires,
            dim=dim,
            device=self.device,
            name=name or gate_name,
            params=gate_params,
        )

    @cached_property
    def CX(self) -> Operator:
        """
        Controlled-X (generalized CNOT) as a standalone dense gate as $CX = CU(X)$ where the target is the generalized X/shift gate.
        """
        return self.CU(self.X, matrix=True)

    @cached_property
    def SWAP(self) -> Operator:
        """
        SWAP gate exchanging two d-dimensional subsystems such that $SWAP|a,b\\rangle = |b,a\\rangle$.
        """
        d = self.dim
        m = torch.zeros((d * d, d * d), dtype=C64, device=self.device)
        for i in range(d):
            for j in range(d):
                row = i * d + j
                col = j * d + i
                m[col, row] = 1.0

        return Operator(m, "SWAP")

    def U(self, matrix: Any, **kwargs: Any) -> Callable[[Any, int, Any], Gate]:
        t = tensorise(matrix, device=self.device)

        name = kwargs.get("name") or "U"

        def factory(dim: Any, wires: int, index: Any, **kwargs: Any) -> Gate:
            """
            Create a Gate instance for given system dimensions/wires and target index set.
            """
            params = kwargs.get("params")

            return Gate(
                t, index, wires, dim, device=self.device, name=name, params=params
            )

        gate = factory
        gate.name = name

        return gate


def _gmr_factory(
    j: int, k: int, type_: str, dim: int, device: str
) -> "Callable[[torch.Tensor], torch.Tensor]":
    """
    Return a closure that builds the GMR unitary matrix from a scalar angle tensor.
    All constant matrices are captured upfront so each call is just `scalar * buffer`.
    """
    d = dim
    C64_ = torch.complex64

    if type_ in ("sym", "asym"):
        a_idx, b_idx = min(j, k), max(j, k)
        I = torch.eye(d, dtype=C64_, device=device)
        E_aa = torch.zeros(d, d, dtype=C64_, device=device)
        E_bb = torch.zeros(d, d, dtype=C64_, device=device)
        E_ab = torch.zeros(d, d, dtype=C64_, device=device)
        E_ba = torch.zeros(d, d, dtype=C64_, device=device)
        E_aa[a_idx, a_idx] = 1.0
        E_bb[b_idx, b_idx] = 1.0
        E_ab[a_idx, b_idx] = 1.0
        E_ba[b_idx, a_idx] = 1.0
        I_rest = I - E_aa - E_bb

        if type_ == "sym":

            def build_fn(angle: torch.Tensor) -> torch.Tensor:
                c = torch.cos(angle / 2).to(dtype=C64_)
                s = torch.sin(angle / 2).to(dtype=C64_)

                return (
                    I_rest + c * E_aa + c * E_bb + (-1j * s) * E_ab + (-1j * s) * E_ba
                )

        else:

            def build_fn(angle: torch.Tensor) -> torch.Tensor:
                c = torch.cos(angle / 2).to(dtype=C64_)
                s = torch.sin(angle / 2).to(dtype=C64_)

                return I_rest + c * E_aa + c * E_bb + (-s) * E_ab + s * E_ba

    else:  # diag
        gen = gell_mann(j, j, d, device=device)

        def build_fn(angle: torch.Tensor) -> torch.Tensor:
            ang = angle.to(dtype=C64_)

            return torch.matrix_exp(-1j * (ang / 2) * gen)

    return build_fn


class VarGate(Gate):
    """
    A parametric gate that recomputes its unitary from a live `nn.Parameter` angle
    on every forward pass, enabling correct gradient flow through the angle parameter.

    Used by `Gategen.GMR` (and `RX`/`RY`/`RZ`) when the supplied angle has `requires_grad`.
    The matrix is never snapshotted: each `_left()` call invokes `_build_fn(self.angle)`.
    """

    def __init__(
        self,
        build_fn: "Callable[[torch.Tensor], torch.Tensor]",
        angle: Any,
        index: List[int],
        wires: int,
        dim: Union[int, List[int]],
        device: str = "cpu",
        name: str = "VarGate",
        params: Optional[list] = None,
    ) -> None:
        if isinstance(angle, torch.Tensor):
            angle_val = angle.detach().float().to(device)
        else:
            angle_val = torch.tensor(float(angle), dtype=torch.float32, device=device)

        with torch.no_grad():
            init_U = build_fn(angle_val.to(dtype=C64))

        super().__init__(
            matrix=init_U,
            index=index,
            wires=wires,
            dim=dim,
            device=device,
            name=name,
            params=params,
        )
        self._build_fn = build_fn
        if isinstance(angle, nn.Parameter):
            object.__setattr__(self, "angle", angle)
        else:
            self.angle = nn.Parameter(angle_val)

    def _left(
        self, tensor: torch.Tensor, U: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if U is None:
            U = self._build_fn(self.angle.to(dtype=C64))

        return super()._left(tensor, U)


class NoisyGate(Gate):
    """
    A per-wire noise channel applied after each gate in `Mode.NOISY` circuits.

    Subclasses Gate to reuse all wire-permutation logic (perm, inv_perm, target_size, etc.).
    The placeholder zeros matrix passed to Gate.__init__ is never used, instead Kraus
    operators are computed live in `forwardd` from `self.param` via differentiable ops
    of the form `scalar(param) * constant_buffer`.

    Supported noise types: "depolarizing", "amplitude_damping", "phase_damping", "pauli".
    """

    SUPPORTED = ("depolarizing", "amplitude_damping", "phase_damping", "pauli", "weyl")

    def __init__(
        self,
        noise_type: str,
        param: Any,
        index: Any,
        wires: int,
        dims: Any,
        device: str = "cpu",
    ) -> None:
        assert (
            noise_type in self.SUPPORTED
        ), f"noise_type must be one of {self.SUPPORTED}"
        idx = index if isinstance(index, list) else [index]
        dim_list = [dims] * wires if isinstance(dims, int) else dims
        d_local = math.prod([dim_list[i] for i in idx])

        super().__init__(
            matrix=torch.zeros(d_local, d_local, dtype=C64),
            index=idx,
            wires=wires,
            dim=dim_list,
            device=device,
            name=f"Noise_{noise_type}",
        )
        self.noise_type = noise_type

        if isinstance(param, nn.Parameter):
            self.param = param
        elif isinstance(param, torch.Tensor) and param.requires_grad:
            self.param = nn.Parameter(param)
        else:
            self.register_buffer(
                "param", torch.as_tensor(param, dtype=torch.float32).to(device)
            )
        self._register_bases()

    def _register_bases(self) -> None:
        """
        Pre-compute constant basis matrices as buffers (no grad, computed once).
        """
        d = self.target_size
        dev = self.device

        if self.noise_type == "amplitude_damping":
            # K_0 = |0><0| + sqrt(1-Y) * (I - |0><0|)
            # K_k = sqrt(Y) * |k-1><k|  for k = 1..d-1
            I00 = torch.zeros(d, d, dtype=C64, device=dev)
            I00[0, 0] = 1.0
            self.register_buffer("_b_I00", I00)
            higher = torch.eye(d, dtype=C64, device=dev) - I00
            self.register_buffer("_b_higher", higher)
            jumps = []
            for k in range(1, d):
                jmp = torch.zeros(d, d, dtype=C64, device=dev)
                jmp[k - 1, k] = 1.0
                jumps.append(jmp)
            self.register_buffer("_b_jumps", torch.stack(jumps))  # (d-1, d, d)

        elif self.noise_type == "phase_damping":
            # K_0 = |0><0| + sqrt(1-λ) * (I - |0><0|)
            # K_k = sqrt(λ) * |k><k|  for k = 1..d-1
            I00 = torch.zeros(d, d, dtype=C64, device=dev)
            I00[0, 0] = 1.0
            self.register_buffer("_b_I00", I00)
            higher = torch.eye(d, dtype=C64, device=dev) - I00
            self.register_buffer("_b_higher", higher)
            projs = []
            for k in range(1, d):
                proj = torch.zeros(d, d, dtype=C64, device=dev)
                proj[k, k] = 1.0
                projs.append(proj)
            self.register_buffer("_b_projs", torch.stack(projs))  # (d-1, d, d)

        elif self.noise_type == "depolarizing":
            if d == 2:
                self.register_buffer("_b_I", torch.eye(2, dtype=C64, device=dev))
                self.register_buffer(
                    "_b_X", torch.tensor([[0, 1], [1, 0]], dtype=C64, device=dev)
                )
                self.register_buffer(
                    "_b_Y", torch.tensor([[0, -1j], [1j, 0]], dtype=C64, device=dev)
                )
                self.register_buffer(
                    "_b_Z", torch.tensor([[1, 0], [0, -1]], dtype=C64, device=dev)
                )
            else:
                # General qudit: use d^2 Gell-Mann matrices (including identity)
                gms = [torch.eye(d, dtype=C64, device=dev)]
                for j in range(d):
                    for k in range(d):
                        if j != k or j < d - 1:
                            gms.append(gell_mann(j, k, d, device=dev))
                # Store as a stacked buffer: shape (d^2, d, d)
                self.register_buffer("_b_gm", torch.stack(gms))

        elif self.noise_type == "pauli":
            assert d == 2, "pauli channel requires d=2 (qubit) wires"
            self.register_buffer("_b_I", torch.eye(2, dtype=C64, device=dev))
            self.register_buffer(
                "_b_X", torch.tensor([[0, 1], [1, 0]], dtype=C64, device=dev)
            )
            self.register_buffer(
                "_b_Y", torch.tensor([[0, -1j], [1j, 0]], dtype=C64, device=dev)
            )
            self.register_buffer(
                "_b_Z", torch.tensor([[1, 0], [0, -1]], dtype=C64, device=dev)
            )

        elif self.noise_type == "weyl":
            # Heisenberg-Weyl displacement operators W_mn = X^m Z^n, (m,n) != (0,0)
            # X = cyclic shift, Z = clock (diagonal phase)
            omega = torch.exp(torch.tensor(2j * math.pi / d, dtype=C64, device=dev))
            X = torch.zeros(d, d, dtype=C64, device=dev)
            for j in range(d):
                X[j, (j - 1) % d] = 1.0
            Z = torch.diag(torch.stack([omega**k for k in range(d)]))
            # build all W_mn for (m,n) in {0..d-1}^2 \ {(0,0)}
            weyl_ops = []
            for m in range(d):
                Xm = torch.linalg.matrix_power(X, m)
                for n in range(d):
                    if m == 0 and n == 0:
                        continue
                    Zn = torch.linalg.matrix_power(Z, n)
                    weyl_ops.append(Xm @ Zn)
            self.register_buffer("_b_weyl", torch.stack(weyl_ops))  # (d²-1, d, d)
            self.register_buffer("_b_I_weyl", torch.eye(d, dtype=C64, device=dev))

    def _kraus_ops(self) -> List[torch.Tensor]:
        """
        Return Kraus matrices as `scalar(param) * constant_buffer`.
        All scalars are differentiable w.r.t. self.param;  the computation graph is preserved.
        """
        if self.noise_type == "amplitude_damping":
            g = self.param.clamp(0, 1).to(C64)
            K0 = self._b_I00 + torch.sqrt(1 - g) * self._b_higher

            return [K0] + list(torch.sqrt(g) * self._b_jumps)

        elif self.noise_type == "phase_damping":
            lam = self.param.clamp(0, 1).to(C64)
            K0 = self._b_I00 + torch.sqrt(1 - lam) * self._b_higher

            return [K0] + list(torch.sqrt(lam) * self._b_projs)

        elif self.noise_type == "depolarizing":
            p = self.param.clamp(0, 1)
            d = self.target_size
            if d == 2:
                r = torch.sqrt(1 - p).to(C64)
                s = torch.sqrt(p / 3).to(C64)

                return [
                    r * self._b_I,
                    s * self._b_X,
                    s * self._b_Y,
                    s * self._b_Z,
                ]
            else:
                # Generalized depolarizing: (1-p)*I + p/(d^2-1) * sum_j G_j rho G_j†
                # Kraus: sqrt(1-p)*I, sqrt(p/(d^2-1))*G_j for j=1..d^2-1
                n_extra = d * d - 1
                r = torch.sqrt(1 - p).to(C64)
                s = torch.sqrt(p / n_extra).to(C64)
                gms = self._b_gm  # shape (d^2, d, d)

                return [r * gms[0]] + list(s * gms[1:])

        elif self.noise_type == "pauli":
            px = self.param[0].clamp(0, 1)
            py = self.param[1].clamp(0, 1)
            pz = self.param[2].clamp(0, 1)
            p_sum = (px + py + pz).clamp(max=1)
            r = torch.sqrt(1 - p_sum).to(C64)

            return [
                r * self._b_I,
                torch.sqrt(px).to(C64) * self._b_X,
                torch.sqrt(py).to(C64) * self._b_Y,
                torch.sqrt(pz).to(C64) * self._b_Z,
            ]

        else:  # weyl
            # param has d²-1 entries (probabilities for each W_mn, (m,n)!=(0,0))
            p = self.param.clamp(0, 1)
            p_sum = p.sum().clamp(max=1)
            r = torch.sqrt(1 - p_sum).to(C64)

            return [r * self._b_I_weyl] + list(
                torch.sqrt(p).to(C64).view(-1, 1, 1) * self._b_weyl
            )

    def forwardd(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Apply the noise channel $\\Phi(\\rho) = \\sum_k K_k \\rho K_k^\\dagger$ to a density matrix.
        Reuses Gate._left with explicit U to avoid full matrix materialization.
        """
        rho_out = torch.zeros_like(rho)
        for K in self._kraus_ops():
            rho_prime = self._left(rho, K)
            rho_prime_T = rho_prime.conj().T
            out_T = self._left(rho_prime_T, K)
            rho_out = rho_out + out_T.conj().T

        return rho_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "NoisyGate only supports density matrix evolution via forwardd()"
        )
