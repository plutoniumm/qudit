from typing import Any, Union, Optional
from functools import cached_property
from ..circuit.gates import Gate
from ..index import State
import numpy as np
import torch as pt

C64 = pt.complex64


class Error(np.ndarray):
    """
    A Kraus/error operator represented as a NumPy ndarray subclass.

    Conceptually, an error operator $E$ acts on a state as $\\rho \mapsto E\\rho E^\dagger$,
    and a noise channel is typically a collection $\{E_k\}$.
    """

    params: dict[str, Any]
    correctable: bool = False
    name: str
    d: int

    def __new__(
        cls,
        d: int,
        O: Optional[np.ndarray] = None,
        name: str = "Err",
        params: Optional[dict[str, Any]] = None,
    ) -> "Error":
        """
        Construct an Error from a matrix $O$ and annotate it with metadata.
        """
        if O is None:
            O = np.zeros((d, d), dtype=complex)

        obj = np.asarray(O).view(cls)
        obj.params = params or {}
        obj.name = name
        obj.d = d

        return obj

    def __array_finalize__(self, obj: Any) -> None:
        if obj is None:
            return
        self.params = getattr(obj, "params", {})
        self.correctable = getattr(obj, "correctable", False)
        self.d = getattr(obj, "d", 0)
        self.name = getattr(obj, "name", "Err")

    def __repr__(self) -> str:
        return f"{self.name}({self.params})"


class Channel:
    """
    A quantum channel represented in Kraus form using localized Gates.

    For Kraus operators $\{E_k\}$, the channel acts as
    $\Phi(\\rho) = \sum_k E_k\,\\rho\,E_k^{\dagger}$.
    """

    correctables: list[Union[int, list[int]]]
    ops: list[list[Gate]]

    def __init__(self, ops: list[list[Gate]]):
        assert (
            isinstance(ops, list) and len(ops) > 0
        ), "ops must be a list of Gate lists"
        self.ops = ops
        self.correctables = []
        first = ops[0][0]
        self.d: int = first.total_dim

    def run(self, rho: Union["State", pt.Tensor, np.ndarray]) -> pt.Tensor:
        """
        Apply the channel in Kraus form: $\\rho \mapsto \sum_k E_k\\rho E_k^{\dagger}$.
        Executes localized gate operations via `forwardd`.
        """
        if hasattr(rho, "isDensity"):
            if not rho.isDensity:
                rho = rho.density()
        else:
            if isinstance(rho, pt.Tensor) and rho.ndim == 1:
                rho = rho.view(-1, 1) @ pt.conj(rho.view(1, -1))
                rho = rho.to(pt.complex64)

        tensor_rho = getattr(rho, "tensor", rho)

        if isinstance(tensor_rho, np.ndarray):
            if tensor_rho.ndim == 1:
                tensor_rho = np.outer(tensor_rho, tensor_rho.conj())
            tensor_rho = pt.from_numpy(tensor_rho)

        device = self.ops[0][0].device if len(self.ops[0]) > 0 else "cpu"
        tensor_rho = tensor_rho.to(device=device, dtype=C64)

        rho_out = pt.zeros_like(tensor_rho)

        for kraus_word in self.ops:
            rho_branch = tensor_rho.clone()

            for g in kraus_word:
                rho_branch = g.forwardd(rho_branch)

            rho_out += rho_branch

        return rho_out

    def correctable(self) -> Any:
        """
        Return the subset(s) of Kraus operators marked as correctable (if any).
        """
        if not self.correctables:
            return []

        c0 = self.correctables[0]

        if isinstance(c0, int):
            idxs = [i for i in self.correctables if isinstance(i, int)]

            return [self.ops[i] for i in idxs]

        if isinstance(c0, list):
            Ek: list[list[list[Gate]]] = []
            sets = [s for s in self.correctables if isinstance(s, list)]
            for s in sets:
                Ek.append([self.ops[i] for i in s])

            return Ek

        return Exception("Please don't change correctables")

    def __getitem__(
        self, key: Union[int, slice]
    ) -> Union[list[Gate], list[list[Gate]]]:
        return self.ops[key]

    def __repr__(self) -> str:
        return f"Channel({len(self.ops)} Kraus operators)"

    def _materialize(self, word: "list[Gate]") -> pt.Tensor:
        """
        Materialize the full $d \\times d$ matrix for a Kraus word by passing
        an identity matrix through each gate's $\\_left()$ call.
        """
        first = self.ops[0][0]
        mat = pt.eye(self.d, dtype=C64, device=first.device)
        for g in word:
            mat = g._left(mat)

        return mat

    @cached_property
    def isTP(self) -> bool:
        """
        Check trace-preservation (TP) via $\sum_k E_k^{\\dagger}E_k = I$ (approx.).
        """
        first = self.ops[0][0]
        total = pt.zeros((self.d, self.d), dtype=C64, device=first.device)
        for word in self.ops:
            Ek = self._materialize(word)
            total += Ek.conj().T @ Ek
        identity = pt.eye(self.d, dtype=C64, device=first.device)

        return bool(pt.allclose(total, identity, atol=1e-8))

    @cached_property
    def isCP(self) -> bool:
        """
        Check complete-positivity (CP) by verifying Choi matrix is PSD.
        """
        J = self.toChoi()
        eig = pt.linalg.eigvalsh(J.to(pt.complex128))

        return bool((eig.real >= -1e-5).all().item())

    @cached_property
    def isCPTP(self) -> bool:
        """
        Check if the channel is CPTP (completely positive and trace preserving).
        """
        return self.isCP and self.isTP

    def toChoi(self) -> pt.Tensor:
        """
        Compute the Choi matrix $J(\Phi) = \sum_{i,j} |i\\rangle\langle j| \otimes \Phi(|i\\rangle\langle j|)$.
        """
        d = self.d
        J = pt.zeros((d * d, d * d), dtype=pt.complex128)
        basis = pt.eye(d, dtype=pt.complex128)
        for i in range(d):
            for j in range(d):
                Eij = pt.outer(basis[i], basis[j])
                J += pt.kron(Eij, self.run(Eij).to(pt.complex128))

        return J

    def toSuperop(self) -> pt.Tensor:
        """
        Compute the superoperator $S$ such that $\mathrm{vec}(\Phi(\\rho)) = S\,\mathrm{vec}(\\rho)$,
        i.e. $S = \sum_k E_k \otimes E_k^*$.
        """
        mats = [self._materialize(word).to(pt.complex128) for word in self.ops]
        d = self.d
        S = pt.zeros((d * d, d * d), dtype=pt.complex128)
        for Ek in mats:
            S += pt.kron(Ek, Ek.conj())

        return S

    def toStinespring(self) -> pt.Tensor:
        """
        Compute a Stinespring isometry $V$ by stacking Kraus operators.

        Constructs $V: \mathbb{C}^d \\to \mathbb{C}^d \otimes \mathbb{C}^r$ with
        $V = \sum_k |k\\rangle \otimes E_k$, represented as a $(dr)\\times d$ matrix.
        """
        K = len(self.ops)
        d = self.d
        r = K
        V = pt.zeros((d * r, d), dtype=pt.complex128)
        for n, O in enumerate(self.ops):
            Ek = self._materialize(O).to(pt.complex128)
            V[n * d : (n + 1) * d, :] = Ek

        return V

    @property
    def Ak(self) -> list[Error]:
        """
        Alias for the Kraus list $\{E_k\}$
        """
        return self.ops


class Multiplex:
    """
    Container class for multiple channels, e.g. for different noise models or parameter regimes.
    Allows us to run multiple channels on a state successively.
    """

    channels: list["Channel"]

    def __init__(self, channels: list[Union["Channel", "Multiplex"]]):
        assert (
            isinstance(channels, list) and len(channels) > 0
        ), "channels must be List[Channel] or List[Multiplex]"

        clist: list["Channel"] = []
        for ch in channels:
            if isinstance(ch, Multiplex):
                clist.extend(ch.channels)
            else:
                clist.append(ch)

        self.channels = clist

    def run(self, rho: Union["State", pt.Tensor, np.ndarray]) -> pt.Tensor:
        """
        Apply channels in sequence: $\\rho \mapsto \Phi_n \circ \cdots \circ \Phi_1(\\rho)$.
        """
        result = rho
        for channel in self.channels:
            result = channel.run(result)

        return result

    def __getitem__(self, key: Union[int, slice]) -> Union["Channel", list["Channel"]]:
        return self.channels[key]

    def __repr__(self) -> str:
        return f"Multiplex({len(self.channels)} channels)"
