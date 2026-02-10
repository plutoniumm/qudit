from typing import List, Optional, Union, Callable, Any, Tuple
from .. import State
import torch.nn as nn
import numpy as np
import torch

C64 = torch.complex64


class Gate:
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
        # Gate ^ Gate
        if isinstance(other, Gate):
            tensor = torch.kron(self.tensor, other.tensor)
            name = f"{self.name} ^ {other.name}"
            params = self.params + other.params
            return Gate(tensor, name, params)

        # Gate ^ Unitary
        if isinstance(other, Unitary):
            tensor = torch.kron(self.tensor, other.matrix())
            name = f"{self.name} ^ {other.name}"
            params = self.params + list(other.params)
            return Gate(tensor, name, params)

        # Gate ^ Tensor
        if isinstance(other, torch.Tensor):
            tensor = torch.kron(self.tensor, other)
            name = f"{self.name} ^ Tensor"
            return Gate(tensor, name)

        # Gate ^ State
        if isinstance(other, State):
            tensor = torch.kron(self.tensor, other.tensor if hasattr(other, "tensor") else other)  # type: ignore[arg-type]
            return State(tensor)  # type: ignore[call-arg]

        raise TypeError("Can only tensor product with Gate, Unitary, Tensor, or State")

    def __rxor__(self, other: Any) -> Union["Gate", Any]:
        if isinstance(other, Unitary):
            tensor = torch.kron(other.matrix(), self.tensor)
            name = f"{other.name} ^ {self.name}"
            params = list(other.params) + list(self.params)
            return Gate(tensor, name, params)

        return NotImplemented

    def __matmul__(self, other: Any) -> Any:
        # Gate @ State
        if isinstance(other, State):
            return State(self.tensor @ (other.tensor if hasattr(other, "tensor") else other))  # type: ignore[call-arg]
        # Gate @ Gate
        elif isinstance(other, Gate):
            tensor = self.tensor @ other.tensor
            name = f"{self.name} @ {other.name}"
            params = self.params + other.params
            return Gate(tensor, name, params)
        # Gate @ Tensor
        elif isinstance(other, torch.Tensor):
            tensor = self.tensor @ other
            name = f"{self.name} @ Tensor"
            return Gate(tensor, name)
        else:
            raise TypeError("Can only apply to State or compose with another Gate")

    def __rmatmul__(self, other: Any) -> Any:
        matmul = getattr(other, "__matmul__", None)

        if callable(matmul):
            return matmul(self)

        raise TypeError(f"Operator '@' not supported between {type(other)} and Gate")


def tensorise(m: Any, device: str = "cpu", dtype: torch.dtype = C64) -> torch.Tensor:
    """
    Convert common array-likes (Tensor/ndarray/list/Gate) into a torch complex tensor.
    """
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


def gell_mann(j: int, k: int, d: int, device: str = "cpu") -> torch.Tensor:
    """
    Return a (generalized) Gell-Mann generator $\lambdaλ_{jk}$ for $SU(d)$.
    """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the embedded unitary to a statevector (reshaping/permuting wires as needed) as $\\rho \\rightarrow U |\psi \\rangle$
        """
        psi = x.view(*self.dims)
        psi = psi.permute(*self.perm)
        psi_flat = psi.reshape(self.target_size, self.rest_size)
        psi_out = self.U @ psi_flat
        current_dims = [self.dims[i] for i in self.perm]
        psi_out = psi_out.view(*current_dims)
        psi_final = psi_out.permute(*self.inv_perm).contiguous()

        return psi_final.view(self.total_dim, 1)

    def forwardd(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Apply the unitary channel to a density matrix as $\\rho \\rightarrow U \\rho U^\dagger$.
        """
        U = self.matrix()
        return U @ rho @ U.conj().T

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
            return State(self.forward(other.tensor if hasattr(other, "tensor") else other))  # type: ignore[call-arg]
        # Gate
        elif isinstance(other, Gate):
            U_full = self.matrix()
            mat = U_full @ tensorise(other.tensor, device=self.device)
            name = f"{self.name} @ {other.name}"
            params = list(self.params) + list(other.params)
            return Gate(mat, name, params)
        # Tensor-like
        elif isinstance(other, torch.Tensor):
            U_full = self.matrix()
            mat = U_full @ other.to(device=self.device, dtype=C64)
            name = f"{self.name} @ Tensor"
            return Gate(mat, name)
        else:
            raise TypeError("Can only apply Unitary to State, Gate, or Tensor")


class Gategen:
    """
    Convenience factory for common qudit gates at a fixed local dimension.
    """

    dim: int
    device: str

    def __init__(self, dim: int = 2, device: str = "cpu"):
        self.dim = dim
        self.device = device

    def asU(
        self,
        m: torch.Tensor,
        index: Union[int, List[int]],
        wires: int,
        dim: Union[int, List[int]],
        name: Optional[str] = None,
        params: Optional[list] = None,
    ) -> Unitary:
        """
        Wrap a target-space matrix as an embedded Unitary acting on given wire indices.
        """
        return Unitary(
            m,
            index=index,  # type: ignore[arg-type]
            wires=wires,
            dim=dim,
            device=self.device,
            name=name or "U",
            params=params,
        )

    @property
    def I(self) -> Gate:
        """
        Identity gate on one qudit such that $I|k\\rangle = |k\\rangle$ for all states $|k\\rangle$.
        """
        return Gate(torch.eye(self.dim, dtype=C64, device=self.device), "I")

    @property
    def H(self) -> Gate:
        """
        Hadamard/DFT gate (H for d=2, discrete Fourier transform for $d$>2) such that $H|k\\rangle = \\frac{1}{\sqrt{d}} \sum_{j=0}^{d-1} \omega^{jk} |j\\rangle$ where $\omega = e^{2\\pi i / d}$.
        """
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
    def X(self) -> Gate:
        """
        Generalized X (cyclic shift) gate as $X|k\\rangle = |k+1 \mod d\\rangle$ (Pauli-X for $d=2$).
        """
        if self.dim == 2:
            m = torch.tensor([[0, 1], [1, 0]], dtype=C64, device=self.device)
        else:
            m = torch.roll(torch.eye(self.dim, dtype=C64, device=self.device), shifts=-1, dims=1)

        return Gate(m, "X")


    @property
    def Z(self) -> Gate:
        """
        Generalized Z (phase) gate: diag(ω^k) as $Z|k\\rangle = \omega^k |k\\rangle$ where $\omega = e^{2\\pi i / d}$ (Pauli-Z for $d=2$).
        """
        d = self.dim
        if d == 2:
            m = torch.tensor([[1, 0], [0, -1]], dtype=C64, device=self.device)
        else:
            w = np.exp(2j * torch.pi / d)
            idx = torch.arange(d, device=self.device)
            m = torch.diag(w**idx)
        return Gate(m, "Z")

    @property
    def Y(self) -> Gate:
        """
        Generalized Y (up to phase), built from Z and X (Pauli-Y for d=2) as $Y = Z X / i$ such that $Y|k\\rangle = -i \omega^k |k+1 \mod d\\rangle$.
        """
        d = self.dim
        if d == 2:
            m = torch.tensor([[0, -1j], [1j, 0]], dtype=C64, device=self.device)
        else:
            m = torch.matmul(self.Z.tensor, self.X.tensor) / 1j
        return Gate(m, "Y")

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
    ) -> Union[Gate, Unitary]:
        """
        Generalized rotation from a Gell-Mann generator (symmetric/asymmetric/diagonal). Gell-Mann gates are described by their type (sym/asym/diag) and the indices j, k specifying the generator.

        We generate each of them as $GMR_{\\text{sym}}(j,k,\\theta) = \\exp(-i \\frac{\\theta}{2} \\lambda_{jk}^{\\text{sym}})$, $GMR_{\\text{asym}}(j,k,\\theta) = \\exp(-i \\frac{\\theta}{2} \\lambda_{jk}^{\\text{asym}})$, and $GMR_{\\text{diag}}(j,j,\\theta) = \\exp(-i \\frac{\\theta}{2} \\lambda_{jj}^{\\text{diag}})$ where $\\lambda_{jk}$ are the Gell-Mann generators.
        """
        if not self.inCircuit(kwargs):
            matrix = True

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
            return Gate(m, gate_name, params=gate_params)

        if not self.inCircuit(kwargs):
            raise TypeError(
                f"{gate_name} missing circuit kwargs (index/wires/dim). "
                f"Call with matrix=True for standalone Gate."
            )

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

    def RX(
        self, angle: Any, *, matrix: bool = False, **kwargs: Any
    ) -> Union[Gate, Unitary]:
        """
        Rotation in the (0,1) symmetric subspace (qubit-like Rx when d=2) as $RX(\\theta) = GMR_{\\text{sym}}(0,1,\\theta)$.
        """
        return self.GMR(0, 1, angle, type="sym", matrix=matrix, **kwargs)

    def RY(
        self, angle: Any, *, matrix: bool = False, **kwargs: Any
    ) -> Union[Gate, Unitary]:
        """
        Rotation in the (0,1) asymmetric subspace (qubit-like Ry when d=2) as $RY(\\theta) = GMR_{\\text{asym}}(0,1,\\theta)$.
        """
        return self.GMR(0, 1, angle, type="asym", matrix=matrix, **kwargs)

    def RZ(
        self, angle: Any, *, matrix: bool = False, **kwargs: Any
    ) -> Union[Gate, Unitary]:
        """
        Diagonal generator rotation (qubit-like Rz when d=2) as $RZ(\\theta) = GMR_{\\text{diag}}(0,0,\\theta)$.
        """
        return self.GMR(0, 0, angle, type="diag", matrix=matrix, **kwargs)

    def CU(
        self, U_target: Any = None, *, matrix: bool = False, **kwargs: Any
    ) -> Union[Gate, Unitary]:
        """
        Controlled-unitary: apply target block when control is in a chosen computational state such that when $U_target$ is a dxd unitary matrix, $CU = |0\\rangle\\langle 0| \\otimes I + |1\\rangle\\langle 1| \\otimes U$ (generalized CNOT for $U=X$ and $d=2$).
        """
        d = self.dim

        U_mat = tensorise(
            U_target.tensor if isinstance(U_target, Gate) else U_target,
            device=self.device,
        )
        if U_mat.shape != (d, d):
            raise ValueError(
                f"U_target must be a ({d},{d}) matrix, got {U_mat.shape}."
            )

        I = torch.eye(d, device=self.device, dtype=C64)
        blocks = [I]
        for k in range(1, d):
            blocks.append(blocks[-1] @ U_mat)

        m = torch.block_diag(*blocks)
        gate_name = "CU"

        target_name = U_target.name if isinstance(U_target, Gate) else None
        gate_params: List[Tuple[str, Any]] = [
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
    def CX(self) -> Gate:
        """
        Controlled-X (generalized CNOT) as a standalone dense gate as $CX = CU(X)$ where the target is the generalized X/shift gate.
        """
        return self.CU(self.X, matrix=True)  # type: ignore[return-value]

    @property
    def SWAP(self) -> Gate:
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
        return Gate(m, "SWAP")

    def U(self, matrix: Any, **kwargs: Any) -> Callable[[Any, int, Any], Unitary]:
        t = tensorise(matrix, device=self.device)

        name = kwargs.get("name") or "U"

        def factory(dim: Any, wires: int, index: Any, **kwargs: Any) -> Unitary:
            """Create a Unitary instance for given system dimensions/wires and target index set."""
            params = kwargs.get("params")
            return Unitary(
                t, index, wires, dim, device=self.device, name=name, params=params
            )

        gate = factory
        gate.name = name  # type: ignore[attr-defined]
        return gate
