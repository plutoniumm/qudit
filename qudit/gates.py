from functools import cache, cached_property as cproperty
from scipy.sparse import csr_matrix, kron
from .index import Gate, Basis, VarGate
from .algebra import Unity, dGellMann
from typing import List, Tuple, Union
import numpy.linalg as LA
import numpy as np

ck = 21

"""
01 = sw
12 = I . sw
23 = I . I . sw1

03 =
sw
I  sw
I  I  sw
I  sw
sw
"""


class Swapper:
    def __init__(self, d: int, width: int, swap: Gate, I: Gate):
        self.d = d
        self.width = width
        self.swap = csr_matrix(swap)
        self.I = csr_matrix(I)

        swaps = [swap]
        op = swap
        for _ in range(width - 2):
            op = I ^ op
            swaps.append(op)

        for i in range(len(swaps)):
            temp = swaps[i]
            rem = width - temp.span
            temp = temp ^ np.eye(d**rem)
            temp.name = temp.name.replace(".U", ".I" * rem)

            swaps[i] = temp
            del temp

        fwd = list(range(len(swaps)))
        bkd = fwd[:-1]
        bkd.reverse()

        prod = 1
        # [0, 1, 2, 1, 0]
        for i in fwd + bkd:
            prod *= swaps[i]

        swaps.append(prod)
        self.swaps = swaps

    def widen(self, gate: Union[Gate, VarGate]) -> List[np.ndarray]:
        w = self.width
        dits = gate.dits  # [0, 2, 3]
        targ = list(range(w))  # [0, 1, 2, 4]

        idle = [d for d in targ if d not in dits]

        swap = self.recepie(dits + idle, targ)
        gate = gate ^ np.eye(gate.d ** (w - gate.span))

        return swap @ gate @ swap.T

    def cycle_decomp(self, arr, tar) -> List[Tuple[int, int]]:
        swaps = []
        for i in range(len(arr)):
            j = arr.index(tar[i])
            while j > i:
                arr[j], arr[j - 1] = arr[j - 1], arr[j]
                swaps.append((j - 1, j))
                j -= 1

        return swaps

    def recepie(self, arr, tar) -> List[Gate]:
        cycles = self.cycle_decomp(arr, tar)
        if not cycles or len(cycles) == 0:
            return np.eye(self.d**self.width)

        gates = 1
        for sw in cycles:
            gates *= self.swaps[sw[0]]

        return Gate(self.d, gates, "SWAP")

    @cache
    def get(self, a: int, b: int) -> Gate:
        if a == b:
            return self.I
        if a > b:
            a, b = b, a

        if a < 0 or b >= self.d:
            raise IndexError(f"Swap indices {a}, {b} out of range [0, {self.d})")

        # literally just multiply the swap gates
        gate = self.swaps[a]
        for i in range(a + 1, b + 1):
            gate = gate @ self.swaps[i]

        return Gate(self.d, gate, f"SWAP({a}, {b})")


class Gategen:
    d: int
    Ket: Basis
    swapper: Swapper

    def __init__(self, d: int, width: int = 2):
        self.d = d
        self.Ket = Basis(d)
        self.swapper = Swapper(self.d, width, self.swap, self.I)

    def create(self, O: np.ndarray = None, name: str = "U"):
        return Gate(self.d, O, name)

    @cproperty
    def X(self) -> Gate:
        O = np.zeros((self.d, self.d))
        O[0, self.d - 1] = 1
        O[1:, 0 : self.d - 1] = np.eye(self.d - 1)
        return Gate(self.d, O, "X")

    @cproperty
    def Y(self) -> Gate:
        O = np.zeros((self.d, self.d), dtype=complex)
        O[0, self.d - 1] = 1j
        O[1:, 0 : self.d - 1] = np.eye(self.d - 1)
        return Gate(self.d, O, "Y")

    @cproperty
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

        F = lambda k: [LA.matrix_power(U, k), self.Ket(k).density()]
        if rev:
            F = lambda k: [self.Ket(k).density(), LA.matrix_power(U, k)]

        gate = [np.kron(*F(k)) for k in range(self.d)]

        name = U.name if U.name else "U"
        gate = Gate(self.d, sum(gate), "C" + name)
        gate.span = 2

        return gate

    @cproperty
    def CX(self) -> Gate:
        return self.CU(self.X, False)

    @cproperty
    def CY(self) -> Gate:
        return self.CU(self.Y, False)

    @cproperty
    def CZ(self) -> Gate:
        return self.CU(self.Z, False)

    # https://www.ijcte.org/vol11/1252-A3006.pdf
    @cproperty
    def swap(self) -> Gate:
        n = self.d
        nn = n * n
        vec = np.arange(nn).reshape(n, n, order="F").flatten()

        P = np.zeros((nn, nn), dtype=int)
        P[np.arange(nn), vec] = 1

        return Gate(self.d, P, "SWAP")

    @cache
    def long_swap(self, a: int, b: int) -> Gate:
        return self.swapper.get(a, b)

    @cproperty
    def S(self):
        w = Unity(self.d)
        O = np.diag([w**j for j in range(self.d)])
        return Gate(self.d, O, "S")

    @cproperty
    def T(self):
        w = Unity(self.d * 2)
        O = np.diag([w**j for j in range(self.d)])
        return Gate(self.d, O, "T")

    @cache
    def P(self, theta: float):
        w = Unity(self.d * 2)
        O = np.diag([w**j for j in range(self.d)])
        return Gate(self.d, O, f"P({theta:.2f})")

    @cproperty
    def H(self) -> Gate:
        O = np.zeros((self.d, self.d), dtype=complex)
        w = Unity(self.d)
        for j in range(self.d):
            for k in range(self.d):
                O[j, k] = w ** (j * k) / np.sqrt(self.d)

        return Gate(self.d, O, "H")

    @cache
    def Rot(self, thetas: List[complex]) -> Gate:
        R = np.eye(self.d)
        for i, theta in enumerate(thetas):
            R = np.exp(-1j * theta * dGellMann(self.d)[i]) @ R

        return Gate(self.d, R, "Rot")

    @cproperty
    def I(self) -> Gate:
        return Gate(self.d, np.eye(self.d), "I")
