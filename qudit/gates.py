from typing import List, Tuple, Callable, Union
from .index import Gate, Basis, Tensor
from .algebra import Unity, dGellMann
import numpy.linalg as LA
import numpy as np


ck = 21

SuperGate = Union[
    Callable[[int, int], Gate],
    Callable[[int], Gate],
    Gate,
]


class Gategen:
    def __init__(self, d: int):
        self.d = d
        self.Ket = Basis(d)

    @property
    def X(self) -> Gate:
        O = np.zeros((self.d, self.d))
        O[0, self.d - 1] = 1
        O[1:, 0 : self.d - 1] = np.eye(self.d - 1)
        return Gate(self.d, O, "X")

    def CU(self, U: Gate, dits: List[int]):
        ctrl = lambda k: self.Ket(k).density()
        targ = lambda k: LA.matrix_power(U, k)
        width = abs(dits[1] - dits[0]) + 1
        """
          CU = Σ_k U^k ⊗ |k><k| (target, ctrl)
          CU = Σ_k |k><k| ⊗ U^k (ctrl, target)

          for everything else we insert I
          Eg: CU(1, 4) = Σ_k |k><k| ⊗ I ⊗ I ⊗ U^k
        """

        gate = []
        for k in range(self.d):
            Op = [ctrl(k)]
            if width > 2:
                Op += [np.eye(self.d * (width - 2))]
            Op += [targ(k)]

            if dits[0] > dits[1]:
                Op.reverse()

            gate.append(Tensor(*Op))

        name = U.name if U.name else "U"
        gate = Gate(self.d, sum(gate), "C" + name)
        gate.dits = dits
        gate.span = 2
        return gate

    def _cu_apply(self, U: Gate, targ: int, ctrl: int = None) -> SuperGate:
        assert isinstance(targ, int) and targ >= 0, f"Target ({targ}) must be 0<=int"

        def gen(ctrl: int) -> Gate:
            assert isinstance(ctrl, int) and ctrl > 0, f"Ctrl ({ctrl}) must be 0<int"
            assert ctrl != targ, f"ctrl{ctrl} == targ{targ}, is not allowed"

            return self.CU(U, (ctrl, targ))

        if ctrl is None:
            return gen
        else:
            return gen(ctrl)

    def CX(self, *args) -> SuperGate:
        return self._cu_apply(self.X, *args)

    def CY(self, *args) -> SuperGate:
        return self._cu_apply(self.Y, *args)

    def CZ(self, *args) -> SuperGate:
        return self._cu_apply(self.Z, *args)

    @property
    def Z(self) -> Gate:
        w = Unity(self.d)
        O = np.diag([w**i for i in range(self.d)])
        return Gate(self.d, O, "Z")

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

    @property
    def _(self) -> Gate:
        return Gate(self.d, np.array([[0]]), "_")
