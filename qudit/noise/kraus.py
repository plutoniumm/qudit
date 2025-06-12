class Channel:
    Ek: list

    def __init__(self, Ek):
        self.Ek = Ek

    def __call__(self, state):
        if not isinstance(state, State):
            raise TypeError("Input must be a State instance")

        result = np.zeros_like(state, dtype=np.complex128)
        for k in self.Ek:
            result += k @ state @ k.conj().T
        return State(result)

    # @property
    # isCP(self) -> bool:
    #     pass

    # @property
    # isTP(self) -> bool:
    #     pass

    # @property
    # isCPTP(self) -> bool:
    #     return self.isCP and self.isTP


from scipy.special._comb import _comb_int as nCr
from .utils import Error
from typing import List
import numpy as np

C128 = np.complex128


class GAD:
    @staticmethod
    def A(order: int, d: int, Y: float):
        obj = np.zeros((d, d), dtype=C128)
        for r in range(order, d):
            obj[r - order][r] = np.sqrt(nCr(r, order) * (1 - Y) ** r - order * Y**order)

        return Error(d, obj, f"A{order}", {"Y": Y, "order": order})

    @staticmethod
    def R(order: int, d: int, Y: float):
        obj = np.zeros((d, d), dtype=C128)
        for r in range(d - order):
            obj[r + order][r] = np.sqrt(
                nCr(d - r - 1, order) * (1 - Y) ** (d - r - order - 1) * Y**order
            )

        return Error(d, obj, f"R{order}", {"Y": Y, "order": order})


paulis = {
    "I": np.array([[1, 0], [0, 1]], dtype=C128),
    "X": np.array([[0, 1], [1, 0]], dtype=C128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=C128),
    "Z": np.array([[1, 0], [0, -1]], dtype=C128),
}


class Pauli:
    @staticmethod
    def X(p: float):
        x = np.sqrt(p) * paulis["X"]
        return Error(2, x, "X", {"p": p})

    @staticmethod
    def Y(p: float):
        y = np.sqrt(p) * paulis["Y"]
        return Error(2, y, "Y", {"p": p})

    @staticmethod
    def Z(p: float):
        z = np.sqrt(p) * paulis["Z"]
        return Error(2, z, "Z", {"p": p})

    @staticmethod
    def I(ps: List[float]):
        assert len(ps) == 2, "Length of ps must be 3"
        p = np.sqrt(1 - np.sum(ps))
        i = p * paulis["I"]

        return Error(2, i, "I", {"p": p})
class Channel:
    Ek: list

    def __init__(self, Ek):
        self.Ek = Ek

    def __call__(self, state):
        if not isinstance(state, State):
            raise TypeError("Input must be a State instance")

        result = np.zeros_like(state, dtype=np.complex128)
        for k in self.Ek:
            result += k @ state @ k.conj().T
        return State(result)

    # @property
    # isCP(self) -> bool:
    #     pass

    # @property
    # isTP(self) -> bool:
    #     pass

    # @property
    # isCPTP(self) -> bool:
    #     return self.isCP and self.isTP
