from more_itertools import distinct_permutations as permut
from .kraus import GAD, Error, Pauli
from typing import List
import numpy as np

C128 = np.complex128
VEC = lambda x: np.array(x, dtype=C128)

def ungroup(lst: List[List[Error]]) -> List[Error]:
    return [item for sublist in lst for item in sublist]

class Ek(np.ndarray):
    def __new__(cls, Ea: List[str], Y: float, p: float):
        assert isinstance(p, float), "p must be a float"
        assert isinstance(Y, float), "Y must be a float"
        # assert p <= Y and p <= 1 and Y <= 1, "p<Y must be in [0, 1]"
        assert p <= 1 and Y <= 1, "p,Y must be in [0, 1]"

        e = np.array(1, dtype=C128).view(cls)

        tot_order = 0
        name_array = []

        paulis = {
            "I": VEC([[1, 0], [0, 1]]),
            "X": VEC([[0, 1], [1, 0]]),
            "Y": VEC([[0, -1j], [1j, 0]]),
            "Z": VEC([[1, 0], [0, -1]]),
            "A0": VEC([[1, 0], [0, np.sqrt(1 - Y)]]),
            "A1": VEC([[0, np.sqrt(Y)], [0, 0]]),
            "R0": VEC([[np.sqrt(1 - Y), 0], [0, 1]]),
            "R1": VEC([[0, 0], [np.sqrt(Y), 0]]),
        }

        _flips = ["X", "Y", "Z", "R0", "R1"]
        _order1 = ["X", "Y", "Z", "A1", "R1"]
        prob = 1.0
        for typ in Ea:
            typ = typ.upper()
            op = paulis[typ]

            p_ = p if typ in _flips else (1 - p)
            name_array.append(typ)

            prob *= np.sqrt(p_)
            tot_order += 1 if typ in _order1 else 0
            e = np.kron(e, op)
        e *= prob
        e.P = prob**2
        e.name = f"{'.'.join(name_array)}"
        e.order = tot_order
        return e

    @property
    def H(self):
        self.conj().T

        if self.name[-1] == "†":
            self.name = self.name[:-1]
        else:
            self.name += "†"

        return self

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.name = getattr(obj, "name", None)
        self.order = getattr(obj, "order", None)
        self.P = getattr(obj, "P", None)

    def __repr__(self):
        return self.name

class Process(Ek):
    def AD_keys(n, order, group: bool = False):
        keys = ["a0", "a1"] * n

        if group:
            error_op = [[] for i in range(order + 1)]
            for comb in permut(keys, n):
                ord = np.sum([int(Em[-1]) for Em in comb])
                if ord <= order:
                    error_op[ord].append(comb)

        else:
            error_op = []
            for comb in permut(keys, n):
                ord = np.sum([int(Em[-1]) for Em in comb])
                if ord <= order:
                    error_op.append(comb)

        return error_op

    def AD(n, order, Y: float = 0.0, group: bool = False) -> list:
        error_op = []
        error_operators = Process.AD_keys(n, order, group)

        if group:
            for set in error_operators:
                e_set = []
                for error in set:
                    e_set.append(Ek(error, Y, 0.0))
                error_op.append(e_set)

        else:
            for error in error_operators:
                error_op.append(Ek(error, Y, 0.0))

        return error_op

    def AD_full(n, Y: float = None):
        keys = ["a0", "a1"] * n
        error_op = []

        for comb in permut(keys, n):
            error_op.append(Ek(comb, Y, 0.0))

        return error_op

    def Pauli_keys(n, order, paulis: list[str] = ["X", "Y", "Z"], group: bool = False):
        keys = (["I"] + paulis) * n
        value = {"I": 0, "X": 1, "Y": 1, "Z": 1, "i": 0, "x": 1, "y": 1, "z": 1}
        error_op = []

        if group:
            error_op = [[] for i in range(order + 1)]
            for comb in permut(keys, n):
                ord = np.sum([value[Em] for Em in comb])
                if ord <= order:
                    error_op[ord].append(comb)

        else:
            error_op = []
            for comb in permut(keys, n):
                ord = np.sum([value[Em] for Em in comb])
                if ord <= order:
                    error_op.append(comb)

        return error_op

    def Pauli_full(
        n, paulis: list[str] = ["X", "Y", "Z"], p: float = 0.0
    ) -> np.ndarray:

        keys = (["I"] + paulis) * n
        error_op = []

        for comb in permut(keys, n):
            error_op.append(Ek(comb, 0.0, p))

        return error_op

    def Pauli(
        n,
        order,
        paulis: list[str] = ["X", "Y", "Z"],
        p: float = 0.0,
        group: bool = False,
    ) -> np.ndarray:

        error_operators = Process.Pauli_keys(n, order, paulis, group)
        error_op = []

        if group:
            for set in error_operators:
                e_set = []
                for error in set:
                    e_set.append(Ek(error, 0.0, p))
                error_op.append(e_set)

        else:
            for error in error_operators:
                error_op.append(Ek(error, 0.0, p))

        return error_op

    def GAD_keys(n, order, keys: list[str] = ["a0", "a1", "r1"], group: bool = False):
        keys = keys * n

        if group:
            error_op = [[] for i in range((order + 1) * 2 - 1)]
            for comb in permut(keys, n):
                # s = 0
                # for term in comb:
                #     s += int(term[-1]) if term  == 'a' else -1*int(term[-1])
                # key = 2*s if s >= 0 else (-2*s) - 1
                # error_op[key].append(comb)
                s = np.sum([int(Em[-1]) for Em in comb])
                if s <= order:
                    if any("r" in i and int(i[-1]) > 0 for i in comb):
                        error_op[2 * s - 1].append(comb)
                    else:
                        error_op[2 * s].append(comb)
        else:
            error_op = []
            for comb in permut(keys, n):
                s = np.sum([int(Em[-1]) for Em in comb])
                if s <= order:
                    error_op.append(comb)

        return error_op

    def GAD(
        n,
        order,
        keys: list[str] = ["a0", "a1", "r1"],
        Y: float = 0.0,
        p: float = 0.0,
        group: bool = False,
    ) -> list:

        error_operators = Process.GAD_keys(n, order, keys, group)
        error_op = []

        if group:
            for set in error_operators:
                e_set = []
                for error in set:
                    e_set.append(Ek(error, Y, p))
                error_op.append(e_set)

        else:
            for error in error_operators:
                error_op.append(Ek(error, Y, p))

        return error_op

    def GAD_full(n, Y: float, p: float):

        keys = ["a0", "a1", "r0", "r1"] * n
        error_op = []

        for comb in permut(keys, n):
            error_op.append(Ek(comb, Y, p))

        return error_op
