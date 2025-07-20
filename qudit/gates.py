from scipy.sparse import kron, eye_array, dok_matrix
from .index import Gate, Basis, VarGate
from .algebra import Unity, dGellMann
from typing import List, Tuple, Union
import numpy.linalg as LA
import numpy as np
import math as ma

ck = 21


class Swapper:
    def __init__(self, d: int, width: int):
        self.width = width
        self.d = d

    def widen(self, gate: Union[Gate, VarGate]) -> List[np.ndarray]:
        w = self.width
        d = self.d
        dits = gate.dits
        idle = [q for q in range(w) if q not in dits]
        perm = dits + idle

        gate = kron(gate, eye_array(d ** (w - len(dits))), format="dok")
        gate = self.permute_in_place(gate, perm)

        return gate

    def permute_in_place(self, mat, pattern):
        nq = len(pattern)
        d = self.d
        dim = d**nq

        new_mat = dok_matrix((dim, dim), dtype=complex)

        for (r, c), val in mat.items():
            r_digits = self._to_digits(r, nq, d)
            c_digits = self._to_digits(c, nq, d)

            r_perm = self._from_digits([r_digits[i] for i in pattern], d)
            c_perm = self._from_digits([c_digits[i] for i in pattern], d)

            new_mat[r_perm, c_perm] = val

        return new_mat.tocsr()

    def _to_digits(self, x, nq, d):
        return [(x // d**i) % d for i in range(nq)]

    def _from_digits(self, digits, d):
        x = 0
        for val in reversed(digits):
            x = x * d + val
        return x

    def cycle_decomp(self, arr, tar) -> List[Tuple[int, int]]:
        swaps = []
        for i in range(len(arr)):
            j = arr.index(tar[i])
            while j > i:
                arr[j], arr[j - 1] = arr[j - 1], arr[j]
                swaps.append((j - 1, j))
                j -= 1

        return swaps


class Gategen:
    d: int
    Ket: Basis
    swapper: Swapper

    def __init__(self, d: int, width: int = 2):
        self.d = d
        self.Ket = Basis(d)
        self.width = width
        self.swapper = Swapper(self.d, width)

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

    def permute(self, pattern):
        nq = len(pattern)
        d = self.d
        w = d**nq

        mat = eye_array(w, format="dok")
        for r in range(w):
            mat[r, r] = 0
            digits = []
            x = r
            for _ in range(nq):
                digits.append(x % d)
                x //= d

            permuted_digits = [digits[j] for j in pattern]

            pr = 0
            for i in reversed(permuted_digits):
                pr = pr * d + i

            mat[pr, r] = 1

        return mat.tocsr()

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
