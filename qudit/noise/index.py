from functools import cached_property
from typing import Any, Union, List, Optional
from ..index import State
import numpy as np


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


def unnull(lst: List[np.ndarray]) -> List[np.ndarray]:
    """
    Filter out (approximately) zero operators.

    Drops any matrix $A$ with $\|A\| \\approx 0$ (elementwise up to tolerance),
    useful to clean Kraus lists.
    """
    return [matrix for matrix in lst if not np.all(np.isclose(matrix, 0, atol=1e-8))]


class Channel:
    """
    A quantum channel represented in Kraus form.

    For Kraus operators $\{E_k\}$, the channel acts as
    $\Phi(\\rho) = \sum_k E_k\,\\rho\,E_k^{\dagger}$.
    """

    # Public attributes (for documentation and light type guidance)
    correctables: list[Union[int, list[int]]]
    ops: list[Error]
    d: int

    def __init__(self, ops: list[Error]):
        assert isinstance(ops, list) and len(ops) > 0, "ops must be List[ops]"

        self.ops = unnull(ops)  # type: ignore[assignment]
        self.d = ops[0].d if isinstance(ops[0], Error) else int(ops[0].shape[0])
        self.correctables = []

    def run(self, rho: Union[State, np.ndarray]) -> np.ndarray:
        """
        Apply the channel in Kraus form: $\\rho \mapsto \sum_k E_k\\rho E_k^{\dagger}$.
        """
        rho_arr: Any = getattr(rho, "tensor", rho)
        result = [O @ rho_arr @ O.conj().T for O in self.ops]

        return np.sum(result, axis=0)

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
            Ek: list[list[Error]] = []
            sets = [s for s in self.correctables if isinstance(s, list)]
            for s in sets:
                Ek.append([self.ops[i] for i in s])
            return Ek

        return Exception("Please don't change correctables")

    def __getitem__(self, key: Union[int, slice]) -> Union[Error, list[Error]]:
        return self.ops[key]

    def __repr__(self) -> str:
        return f"Channel({len(self.ops)} ops)"

    @cached_property
    def isTP(self) -> bool:
        """
        Check trace-preservation (TP) via $\sum_k E_k^{\dagger}E_k = I$ (approx.).
        """
        ti = [np.trace(O.conj().T @ O) for O in self.ops]

        return bool(np.isclose(sum(ti), 1.0))

    @cached_property
    def isCP(self) -> bool:
        """
        Check complete-positivity (CP) by verifying Choi matrix is PSD.
        """
        J = self.toChoi()
        eig = np.linalg.eigvalsh(J)
        return bool(np.all(eig >= -1e-8))

    @cached_property
    def isCPTP(self) -> bool:
        """
        Check if the channel is CPTP (completely positive and trace preserving).
        """
        return self.isCP and self.isTP

    def toChoi(self) -> np.ndarray:
        """
        Compute the Choi matrix $J(\Phi) = \sum_{i,j} |i\\rangle\langle j| \otimes \Phi(|i\\rangle\langle j|)$.
        """
        d = self.d
        J = np.zeros((d * d, d * d), dtype=complex)
        basis = np.eye(d, dtype=complex)
        for i in range(d):
            for j in range(d):
                Eij = np.outer(basis[:, i], basis[:, j].conj())
                PhiE = self.run(Eij)
                J += np.kron(Eij, PhiE)
        return J

    def toSuperop(self) -> np.ndarray:
        """
        Compute the superoperator $S$ such that $\mathrm{vec}(\Phi(\\rho)) = S\,\mathrm{vec}(\\rho)$.
        """
        d = self.d
        S = np.zeros((d * d, d * d), dtype=complex)
        I = np.eye(d, dtype=complex)
        for k in range(d * d):
            ek = I.flatten()[k]
            E = ek.reshape(d, d)
            vecPhi = self.run(E).flatten()
            S[:, k] = vecPhi
        return S

    def toStinespring(self) -> np.ndarray:
        """
        Compute a Stinespring isometry $V$ by stacking Kraus operators.

        Constructs $V: \mathbb{C}^d \\to \mathbb{C}^d \otimes \mathbb{C}^r$ with
        $V = \sum_k |k\\rangle \otimes E_k$, represented as a $(dr)\\times d$ matrix.
        """
        K = len(self.ops)
        d = self.d
        r = K
        V = np.zeros((d * r, d), dtype=complex)
        for n, O in enumerate(self.ops):
            V[n * d : (n + 1) * d, :] = O
        return V

    @property
    def Ak(self) -> list[Error]:
        """
        Alias for the Kraus list $\{E_k\}$"""
        return self.ops
