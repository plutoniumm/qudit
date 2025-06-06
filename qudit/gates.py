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

def C_Gate(d: int, Ket, U: Gate, dits: List[int]) -> Gate:
    if len(set(dits)) != len(dits):
        raise ValueError(f"Dits must be unique, got: {dits}")
    for dit in dits:
        assert isinstance(dit, int) and dit > 0, f"Dit: {dit} must be 0<int"

    ctrl = lambda k: Ket(k).density()
    targ = lambda k: LA.matrix_power(U, k)
    width = abs(dits[1] - dits[0]) + 1
    """
      CU = Σ_k U^k ⊗ |k><k| (target, ctrl)
      CU = Σ_k |k><k| ⊗ U^k (ctrl, target)

      for everything else we insert I
      Eg: CU(1, 4) = Σ_k |k><k| ⊗ I ⊗ I ⊗ U^k
    """

    gate = []
    for k in range(d):
        Op = [ctrl(k)]
        if width > 2:
            Op += [np.eye(d * (width - 2))]
        Op += [targ(k)]

        if dits[0] > dits[1]:
            Op.reverse()

        gate.append(Tensor(*Op))

    name = U.name if U.name else "U"
    gate = Gate(d, sum(gate), "C" + name)
    gate.dits = dits
    gate.span = 2
    return gate

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

    def CU(self, U: Gate, dits: List[int]) -> SuperGate:
        if not isinstance(dits, list):
          raise TypeError(f"dits must be a list, got {type(dits)}")

        if len(dits) == 2:
            return C_Gate(self.d, self.Ket, U, dits)

        def gen(ctrl: int) -> Gate:
            return C_Gate(self.d, self.Ket, U, [ctrl, dits[0]])

        return gen

    def CX(self, *args) -> SuperGate:
        return self.CU(self.X, *args)

    def CY(self, *args) -> SuperGate:
        return self.CU(self.Y, *args)

    def CZ(self, *args) -> SuperGate:
        return self.CU(self.Z, *args)

    @property
    def Z(self):
     O = np.diag([w(self.d)**i for i in range(self.d)])
     return Gate(self.d, O, "Z")


    @property
    def S(self):
     omega_s = np.exp(2j * np.pi / (2 * self.d))
     O = np.diag([omega_s**j for j in range(self.d)])
     return Gate(self.d, O, "S")

    @property
    def T(self):
       omega_t = np.exp(2j * np.pi / (4 * self.d))
       O = np.diag([omega_t**j for j in range(self.d)])
       return Gate(self.d, O, "T")

    def P(self, theta: float):
      omega = np.exp(1j * theta * np.pi / self.d)
      O = np.diag([omega**j for j in range(self.d)])
      return Gate(self.d, O, f"P({theta})")

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
