from typing import List, Union
import numpy.linalg as LA
import numpy as np

"""
  B = Basis(3) will create a qutrit basis
  so B("111") will return State for |111>
  or B(1, 2, 0) will return State for |120>
"""


class Basis:
    d: int
    span: int = -1

    def __init__(self, d: int):
        self.d = d

    def __call__(self, *args: Union[List[int], str, int]) -> "State":
        if len(args) == 1 and isinstance(args[0], str):
            args = [int(i) for i in args[0]]

        basis = np.eye(self.d, dtype=np.complex128)
        prod = 1
        for ket in args:
            if ket < 0 or ket >= self.d:
                raise ValueError(f"Index {ket} out of bounds for dimension {self.d}")
            prod = np.kron(prod, basis[ket])

        return State(prod)


class State(np.ndarray):

    def __new__(cls, d: Union[np.ndarray, List, "State"]):
        arr = np.asarray(d, dtype=np.complex128)
        if arr.ndim == 1:
            arr /= np.linalg.norm(arr)
        elif arr.ndim == 2:
            if arr.shape[0] != arr.shape[1]:
                raise ValueError("Density matrix must be square")
            arr /= np.trace(arr).real
        else:
            raise ValueError("Input must be 1D (vector) or 2D (density matrix)")

        obj = arr.view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    @property
    def isDensity(self) -> bool:
        return self.ndim == 1

    def isPure(self) -> bool:
        if not self.isDensity:
            return True
        else:
            tr = np.trace(self**2).real
            return np.isclose(tr, 1.0)

    @property
    def d(self):
        return self.shape[0]

    def norm(self) -> "State":
        if self.isDensity:
            return State(self / np.linalg.norm(self))
        else:
            return State(self / self.trace)

    def density(self) -> "State":
        if self.isDensity:
            return State(np.outer(self, self.conj()))

        return self

    @property
    def H(self) -> "State":
        return State(self.conj().T)

    def __xor__(self, other: "State") -> "State":
        return State(np.kron(self, other))

    @property
    def trace(self) -> float:
        if not self.isDensity:
            raise ValueError("Trace is only defined for density matrices")

        return np.trace(self).real

    def proj(self) -> "State":
        if not self.isDensity:
            return self.density()

        evals, evecs = LA.eig(self)
        matrix = sum(
            [
                np.outer(evecs[:, i], evecs[:, i].conj())
                for i in range(len(evals))
                if np.abs(evals[i]) > 1e-8
            ]
        )
        return State(matrix)

    def oproj(self) -> "State":
        proj = self.proj()
        perp = np.eye(proj.shape[0]) - proj

        return State(perp)
