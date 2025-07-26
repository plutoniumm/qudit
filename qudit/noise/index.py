from functools import cached_property
from typing import Any, Union, List
from ..index import State
import numpy as np


class Error(np.ndarray):
    params: dict[str, Any]
    correctable: bool = False
    name: str
    d: int

    def __new__(cls, d: int, O: np.ndarray = None, name: str = "Err", params={}):
        obj = np.asarray(O).view(cls)
        obj.params = params
        obj.name = name
        obj.d = d

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.params = getattr(obj, "params", {})
        self.correctable = getattr(obj, "correctable", False)
        self.d = getattr(obj, "d", 0)
        self.name = getattr(obj, "name", "Err")

    def __repr__(self):
        print(f"Error: {self.name} with params {self.params}")
        return f"{self.name}({self.params})"


def unnull(lst: List[np.ndarray]) -> List[np.ndarray]:
    return [matrix for matrix in lst if not np.all(np.isclose(matrix, 0, atol=1e-8))]


class Channel:
    correctables: list[Union[int, list[int]]] = []
    ops: list[Error]
    d: int

    def __init__(self, ops: list[Error]):
        assert isinstance(ops, list) and len(ops) > 0, "ops must be List[ops]"

        self.ops = unnull(ops)
        self.d = ops[0].d if isinstance(ops[0], Error) else ops[0].shape[0]
        self.correctables = []

    def run(self, rho: Union[State, np.ndarray]) -> np.ndarray:
        result = [O @ rho @ O.conj().T for O in self.ops]

        return np.sum(result, axis=0)

    def correctable(self):
        if len(self.correctables) == 0:
            return []
        c0 = self.correctables[0]

        if isinstance(c0, int):
            return [self.ops[i] for i in self.correctables]

        if isinstance(c0, list):
            Ek = []
            for set in self.correctables:
                Ek.append([self.ops[i] for i in set])
            return Ek

        return Exception("Please don't change correctables")

    def __getitem__(self, key: Union[int, slice]) -> Union[Error, list[Error]]:
        return self.ops[key]

    def __repr__(self):
        return f"Channel({len(self.ops)} ops)"

    @cached_property
    def isTP(self) -> bool:
        ti = [np.trace(O.conj().T @ O) for O in self.ops]

        return np.isclose(sum(ti), 1.0)

    @cached_property
    def isCP(self) -> bool:
        J = self.toChoi()
        eig = np.linalg.eigvalsh(J)
        return np.all(eig >= -1e-8)

    @cached_property
    def isCPTP(self) -> bool:
        return self.isCP and self.isTP

    def toChoi(self) -> np.ndarray:
        # J = sum_{i,j} |i⟩⟨j| ⊗ Φ(|i⟩⟨j|)
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
        # S acting on vec(ρ): vec(Φ(ρ)) = S · vec(ρ)
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
        # Build isometry V: C^d → C^d ⊗ C^r with minimal r (#Kraus ops)
        K = len(self.ops)
        d = self.d
        r = K
        V = np.zeros((d * r, d), dtype=complex)
        for n, O in enumerate(self.ops):
            V[n * d : (n + 1) * d, :] = O
        return V

    # Adding the correctable set property, we can use Channel as the class for error operators
    @property
    def Ak(self) -> list[Error]:
        return self.ops
