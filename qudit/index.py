from sympy import SparseMatrix as Matrix, zeros, eye, simplify
from sympy.physics.quantum import TensorProduct
from typing import List, Union
import numpy.linalg as LA
import numpy as np
import math as ma

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


class Gate(np.ndarray):
    wires: List[int]
    name: str = ""
    vqc: bool
    span: int
    d: int

    def __new__(
        cls, d: int, O: np.ndarray = None, name: str = "U", wires: List[int] = []
    ):
        if isinstance(O, Matrix):
            return VarGate(d, O, name)

        obj = np.asarray(O, dtype=np.complex64).view(cls)

        obj.span = round(ma.log(O.shape[0], d))
        obj.name = name if name else f"U({d})"
        obj.d = d
        obj.wires = wires
        obj.vqc = False
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
        self.wires = getattr(obj, "wires", [])

    def __xor__(self, other: "Gate") -> "Gate":
        name = f"{self.name}.{getattr(other, 'name', 'U')}"

        return Gate(self.d, np.kron(self, other), name)

    def isUnitary(self):
        return np.allclose(self @ self.H, np.eye(self.shape[0]))

    def isHermitian(self):
        return np.allclose(self, self.H)


class VarGate(Matrix):
    def __new__(
        cls, d: int, O: np.ndarray = None, name: str = "U", wires: List[int] = []
    ):
        if O is None:
            raise ValueError("This part is reachable too. deal with it")
        else:
            mat = Matrix(O)
            mat.span = int(np.log(O.shape[0]) / np.log(d))
        # endif

        mat.name = name if name else f"VarGate({d})"
        mat.d = d
        mat.wires = []
        mat.vqc = True

        if len(wires) > 0:
            span = max(wires) - min(wires) + 1
            if span != mat.span:
                raise ValueError(f"Got span: {span}, expected span: {mat.span}")

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
