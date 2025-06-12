from sympy import SparseMatrix as Matrix, zeros, eye, simplify
from sympy.physics.quantum import TensorProduct
from typing import List, Union
import numpy.linalg as LA
from uuid import uuid4
import numpy as np
import math as ma

"""
  B = Basis(3) will create a qutrit basis
  so B("111") will return State for |111>
  or B(1, 2, 0) will return State for |120>
"""


def ID() -> str:
    return str(uuid4()).split("-")[0]


class Basis:
    d: int = None
    span: int = None

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
            arr = arr / np.linalg.norm(arr)
        elif arr.ndim == 2:
            if arr.shape[0] != arr.shape[1]:
                raise ValueError("Density matrix must be square")
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


class Gate(np.ndarray):
    dits: List[int]
    id: str = None
    name: str = ""
    vqc: bool
    span: int
    d: int

    def __new__(
        cls, d: int, O: np.ndarray = None, name: str = None, dits: List[int] = []
    ):
        if isinstance(O, Matrix):
            return VarGate(d, O, name)

        if O is None:
            raise ValueError("Gate must be initialized with a matrix or None")
            obj = np.zeros((d, d), dtype=complex).view(cls)
            obj.span = 1
        else:
            obj = np.asarray(O, dtype=complex).view(cls)
            obj.span = int(ma.log(len(O[0]), d))
        # endif

        obj.name = name if name else f"Gate({d})"
        obj.d = d
        obj.dits = dits
        obj.vqc = False

        if len(dits) > 0:
            span = max(dits) - min(dits) + 1
            if span != obj.span:
                raise ValueError(f"Got span: {span}, expected span: {obj.span}")

        obj.id = ID()
        return obj

    @property
    def H(self):
        return self.conj().T

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.d = getattr(obj, "d", 0)
        self.span = getattr(obj, "span", 0)
        self.name = getattr(obj, "name", "Gate")
        self.vqc = getattr(obj, "vqc", False)
        self.dits = getattr(obj, "dits", [])

    def __xor__(self, other: "Gate") -> "Gate":
        return Gate(self.d, np.kron(self, other), f"{self.name}.{other.name}")

    def isUnitary(self):
        return np.allclose(self @ self.H, np.eye(self.shape[0]))

    def isHermitian(self):
        return np.allclose(self, self.H)


class VarGate(Matrix):
    def __new__(
        cls, d: int, O: np.ndarray = None, name: str = None, dits: List[int] = []
    ):
        if O is None:
            mat = Matrix(np.zeros((d, d), dtype=complex).view(cls))
            mat.span = 1
        else:
            mat = Matrix(O)
            mat.span = int(np.log(O.shape[0]) / np.log(d))
        # endif

        mat.name = name if name else f"VarGate({d})"
        mat.d = d
        mat.dits = []
        mat.vqc = True

        if len(dits) > 0:
            span = max(dits) - min(dits) + 1
            if span != mat.span:
                raise ValueError(f"Got span: {span}, expected span: {mat.span}")

        mat.id = ID()
        return mat

    @property
    def H(self):
        return self.conjugate().T

    def __xor__(self, other: "VarGate") -> "VarGate":
        kron = TensorProduct(self, other)
        return VarGate(self.d, kron, f"{self.name}.{other.name}")

    def isUnitary(self):
        return simplify(self * self.H) == eye(self.shape[0])

    def isHermitian(self):
        return simplify(self - self.H) == zeros(*self.shape)
