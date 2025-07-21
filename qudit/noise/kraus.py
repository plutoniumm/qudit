from scipy.special._comb import _comb_int as nCr
from .index import Error, Channel, isSquare
from typing import List
from .. import State
import numpy as np

C128 = np.complex128


class GAD:
    @staticmethod
    def A(order: int, d: int, Y: float):
        k = order
        assert isinstance(k, int) and k > 0, "d must be int>0"
        assert isinstance(d, int) and d > 0, "d must be int>0"

        obj = np.zeros((d, d), dtype=C128)
        for r in range(k, d):
            a, b = (r - k) / 2, k / 2
            obj[r - k][r] = np.sqrt(nCr(r, k) * (1 - Y) ** a * Y**b)

        return Error(d, obj, f"A{k}", {"Y": Y, "k": k})

    @staticmethod
    def R(k: int, d: int, Y: float):
        obj = np.zeros((d, d), dtype=C128)
        for r in range(d - k):
            a, b = (d - r - k - 1) / 2, k / 2
            obj[r + k][r] = np.sqrt(nCr(d - r - 1, k) * (1 - Y) ** a * Y**b)

        return Error(d, obj, f"R{k}", {"Y": Y, "k": k})


class Pauli:
    @staticmethod
    def X(p: float):
        x = np.sqrt(p) * np.array([[0, 1], [1, 0]], dtype=C128)
        return Error(2, x, "X", {"p": p})

    @staticmethod
    def Y(p: float):
        y = np.sqrt(p) * np.array([[0, -1j], [1j, 0]], dtype=C128)
        return Error(2, y, "Y", {"p": p})

    @staticmethod
    def Z(p: float):
        z = np.sqrt(p) * np.array([[1, 0], [0, -1]], dtype=C128)
        return Error(2, z, "Z", {"p": p})

    @staticmethod
    def I(ps: List[float]):
        assert len(ps) == 2, "Length of ps must be 3"
        p = np.sqrt(1 - np.sum(ps))
        i = p * np.array([[1, 0], [0, 1]], dtype=C128)

        return Error(2, i, "I", {"p": p})
