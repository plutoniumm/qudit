from functools import cached_property, lru_cache
from typing import Any, Union, List
from ..index import Gate, State
import numpy as np


def isSquare(i: Union[np.ndarray, List[np.ndarray]]):
    if isinstance(list):
        return all([isSquare(j) for j in i])

    return i.ndim == 2 and i.shape[0] == i.shape[1]


class Error(Gate):
    params: dict[str, Any]

    def __new__(cls, d: int, O: np.ndarray = None, name: str = "Err", params={}):
        obj = super().__new__(cls, d, O, name)
        obj.params = params
        return obj


class Channel:
    ops: list[Error]
    d: int

    def __init__(self, ops: list[Error]):
        assert isinstance(ops, list) and len(ops) > 0, "ops must be List[ops]"
        assert isSquare(ops), "Kraus ops must be square"
        self.ops = ops
        self.d = ops[0].d if isinstance(ops[0], Error) else ops[0].shape[0]

    @lru_cache
    def run(self, rho: Union[State, np.ndarray]) -> np.ndarray:
        result = [O @ rho @ O.conj().T for O in self.ops]

        return sum(result)

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
