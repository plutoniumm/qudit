from .algebra import Unity, dGellMann
from typing import List, Union
from .index import Gate, Basis
import numpy.linalg as LA
from .utils import Tensor
import numpy as np

ck = 21


class Gategen:
    def __init__(self, d: int):
        self.d = d
        self.Ket = Basis(d)

    def create(self, O: np.ndarray = None, name: str = None):
        return Gate(self.d, O, name)

    @property
    def X(self) -> Gate:
        O = np.zeros((self.d, self.d))
        O[0, self.d - 1] = 1
        O[1:, 0 : self.d - 1] = np.eye(self.d - 1)
        return Gate(self.d, O, "X")

    @property
    def Y(self) -> Gate:
        O = np.zeros((self.d, self.d), dtype=complex)
        O[0, self.d - 1] = 1j
        O[1:, 0 : self.d - 1] = np.eye(self.d - 1)
        return Gate(self.d, O, "Y")

    @property
    def Z(self) -> Gate:
        w = Unity(self.d)
        O = np.diag([w**i for i in range(self.d)])
        return Gate(self.d, O, "Z")

    def CU(self, U: Gate, rev=False) -> Gate:
        """
        CU = Σ_k U^k ⊗ |k><k| (target, ctrl)
        CU = Σ_k |k><k| ⊗ U^k (ctrl, target)

        for everything else we insert I
        Eg: CU(1, 4) = Σ_k |k><k| ⊗ I ⊗ I ⊗ U^k
        """

        F = lambda k: [self.Ket(k).density(), LA.matrix_power(U, k)]
        if rev:
            F = lambda k: [LA.matrix_power(U, k), self.Ket(k).density()]

        gate = [np.kron(*F(k)) for k in range(self.d)]

        name = U.name if U.name else "U"
        gate = Gate(self.d, sum(gate), "C" + name)
        gate.span = 2

        return gate

    @property
    def CX(self) -> Gate:
        return self.CU(self.X, False)

    @property
    def CY(self) -> Gate:
        return self.CU(self.Y, False)

    @property
    def CZ(self) -> Gate:
        return self.CU(self.Z, False)

    @property
    def swap(self) -> Gate:
        cx, xc = self.CU(self.X, False), self.CU(self.X, True)

        return Gate(self.d, cx @ xc @ cx, "sw")

    def long_swap(self, a: int, b: int, width: int) -> Gate:
        sw = self.swap
        if a > b:
            a, b = b, a
        assert a <= width and b <= width, f"Index out of bounds ({width}): {a}, {b}"
        assert a >= 0 and b >= 0, f"Negative index not allowed: {a}, {b}"
        assert a != b, f"Cannot swap same index: {a}, {b}"

        gates = []
        for i in range(a, b):
            if a <= i < b:
                mat = [self.I] * (width - 1)
                mat[i] = sw
                mat = Tensor(*mat)
                gates.append(mat)

        prod = gates[0]
        for g in gates[1:]:
            prod = prod @ g

        return Gate(self.d, prod, f"SWAP({a}, {b})[{width}]")

    @property
    def S(self):
        w = Unity(self.d)
        O = np.diag([w**j for j in range(self.d)])
        return Gate(self.d, O, "S")

    @property
    def T(self):
        w = Unity(self.d * 2)
        O = np.diag([w**j for j in range(self.d)])
        return Gate(self.d, O, "T")

    def P(self, theta: float):
        w = Unity(self.d * 2)
        O = np.diag([w**j for j in range(self.d)])
        return Gate(self.d, O, f"P({theta:.2f})")

    @property
    def H(self) -> Gate:
        O = np.zeros((self.d, self.d), dtype=complex)
        w = Unity(self.d)
        for j in range(self.d):
            for k in range(self.d):
                O[j, k] = w ** (j * k) / np.sqrt(self.d)

        return Gate(self.d, O, "H")

    def Rot(self, thetas: List[complex]) -> Gate:
        R = np.eye(self.d)
        for i, theta in enumerate(thetas):
            R = np.exp(-1j * theta * dGellMann(self.d)[i]) @ R

        return Gate(self.d, R, "Rot")

    @property
    def I(self) -> Gate:
        return Gate(self.d, np.eye(self.d), "I")
