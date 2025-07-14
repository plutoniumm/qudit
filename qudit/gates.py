from scipy.sparse import csr_matrix, kron, eye_array
from .index import Gate, Basis, VarGate
from .algebra import Unity, dGellMann
from typing import List, Tuple, Union
import numpy.linalg as LA
import numpy as np
import math as ma

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

        sw = csr_matrix(swap)
        I = csr_matrix(I)

        swaps = [sw]
        for _ in range(width - 2):
            swaps.append(kron(I, swaps[-1], format="csr"))

        for i in range(len(swaps)):
            temp = swaps[i] # SW, I.SW, I.I.SW...
            rem = width - round(ma.log(temp.shape[0], d))
            temp = kron(temp, eye_array(d**rem), format="csr")
            swaps[i] = temp

            assert temp.shape[0] == d**width, f"Swapper {i} has wrong shape {temp.shape} != {d**width}"
            del temp

        fwd = list(range(len(swaps)))
        bkd = fwd[:-1]
        bkd.reverse()

        prod = 1
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
        rem = w - gate.span
        gate = kron(gate, eye_array(gate.d ** rem), format="csr")

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
            return eye_array(self.d**self.width, format='csr')

        gates = 1
        for sw in cycles:
            gates *= self.swaps[sw[0]]

        return gates


class Gategen:
    d: int
    Ket: Basis
    swapper: Swapper

    def __init__(self, d: int, width: int = 2):
        self.d = d
        self.Ket = Basis(d)
        self.swapper = Swapper(self.d, width, self.SWAP, self.I)

    def create(self, O: np.ndarray = None, name: str = "U"):
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
        CU = Σ_k U^k ⊗ |k⟩⟨k| (target, ctrl)
        CU = Σ_k |k⟩⟨k| ⊗ U^k (ctrl, target)

        for everything else we insert I
        Eg: CU(1, 4) = Σ_k |k⟩⟨k| ⊗ I ⊗ I ⊗ U^k
        """

        F = lambda k: [LA.matrix_power(U, k), self.Ket(k).density()]
        if rev:
            F = lambda k: [self.Ket(k).density(), LA.matrix_power(U, k)]

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

    # https://www.ijcte.org/vol11/1252-A3006.pdf
    @property
    def SWAP(self) -> Gate:
        n = self.d
        nn = n * n
        vec = np.arange(nn).reshape(n, n, order="F").flatten()

        P = np.zeros((nn, nn), dtype=int)
        P[np.arange(nn), vec] = 1

        return Gate(self.d, P, "SWAP")

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
