from .kraus import GAD, Error, Pauli
from itertools import permutations
from .index import Channel, Error
from typing import List
import numpy as np

C128 = np.complex128

def permut(lst: List[str], n: int) -> List[List[str]]:

    if n > len(lst):
        raise ValueError("n must be less than or equal to the length of lst")

    return [list(p) for p in set(permutations(lst, n))]

def mkron(args):
    result = args[0]
    for i in range(1, len(args)):
        result = np.kron(result, args[i])
    return result


def ungroup(lst: List[List[Error]]) -> List[Error]:
    return [item for sublist in lst for item in sublist]


class Process:

    @staticmethod
    def GAD(
        d: int, n: int, Y: float, p: float, order: int = 1, group: bool = False
    ) -> Channel:
        assert isinstance(p, float), "p must be a float"
        assert isinstance(Y, float), "Y must be a float"
        assert p <= 1 and Y <= 1, "p,Y must be in [0, 1]"

        def _op_gen(error_word) -> Error:
            temp = []
            for tag in error_word:
                order = int(tag[-1])
                if "a" in tag:
                    temp.append(GAD.A(order, d, Y, p))
                elif "r" in tag:
                    temp.append(GAD.R(order, d, Y, p))
                else:
                    raise ValueError(f"Unknown tag {tag} in error word {error_word}")
            return mkron(temp)

        keys = list(permut(["a0", "a1", "r0", "r1"] * n, n))
        Ak = [_op_gen(key) for key in keys]

        Ek = [[] for _ in range((order + 1) * 2 - 1)]
        for key in keys:
            s = np.sum([int(Em[-1]) for Em in key])
            if s <= order and not np.all(np.isclose(Ak[keys.index(key)], 0, atol=1e-8)):

                if any("r" in i and int(i[-1]) > 0 for i in key):
                    Ek[2 * s - 1].append(keys.index(key))
                else:
                    Ek[2 * s - 0].append(keys.index(key))

        op_ch = Channel(Ak)
        op_ch.correctables = Ek if group else ungroup(Ek)

        return op_ch

    @staticmethod
    def AD(d: int, n: int, Y: float, order: int = 1, group: bool = False) -> Channel:
        assert isinstance(Y, float), "Y must be a float"
        assert Y <= 1, "Y must be in [0, 1]"

        def _op_gen(error_word) -> Error:
            individual = [GAD.A(int(tag[-1]), d, Y) for tag in error_word]
            return mkron(individual)

        keys = list(permut(["a0", "a1"] * n, n))
        Ak = [_op_gen(key) for key in keys]

        Ek = [[] for _ in range(order + 1)]
        for key in keys:
            s = np.sum([int(Em[-1]) for Em in key])
            if s <= order and not np.all(np.isclose(Ak[keys.index(key)], 0, atol=1e-8)):
                Ek[s].append(keys.index(key))

        op_ch = Channel(Ak)
        op_ch.correctables = Ek if group else ungroup(Ek)

        return op_ch

    @staticmethod
    def Pauli(
        n: int,
        paulis: list[str] = ["X", "Y", "Z"],
        p=[0.0, 0.0, 0.0],
        order: int = 1,
        group: bool = False,
    ) -> Channel:
        funcs = {
            "I": Pauli.I,
            "X": lambda p: Pauli.X(p[0]),
            "Y": lambda p: Pauli.Y(p[1]),
            "Z": lambda p: Pauli.Z(p[2]),
        }
        weight = {"I": 0, "X": 1, "Y": 1, "Z": 1}

        def _op_gen(word) -> Error:
            return mkron([funcs[gate](p) for gate in word])

        keys = permut((["I"] + paulis) * n, n)
        Ak = [_op_gen(key) for key in keys]

        Ek = [[] for _ in range(order + 1)]
        for key in keys:
            s = np.sum([weight[i] for i in key])
            if s <= order and not np.all(np.isclose(Ak[keys.index(key)], 0, atol=1e-8)):
                Ek[s].append(keys.index(key))

        op_ch = Channel(Ak)
        op_ch.correctables = Ek if group else ungroup(Ek)

        return op_ch
