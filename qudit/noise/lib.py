from more_itertools import distinct_permutations as permut
from .kraus import GAD, Error, Pauli
from typing import List
import numpy as np

np.set_printoptions(suppress=True)

C128 = np.complex128

def ungroup(lst: List[List[Error]]) -> List[Error]:
    return [item for sublist in lst for item in sublist]


class Krauser:
    @staticmethod
    def AD(n, order, Y=0.0, group=False):
        keys = ["A0", "A1"] * n
        error_op = [[] for _ in range(order + 1)]
        for comb in permut(keys, n):
            ord = sum(int(x[-1]) for x in comb)
            if ord <= order:
                error_op[ord].append(comb)

        def gen(e): return GAD.A(int(e[-1]), 2, Y)
        ops = [[gen(x) for x in grp] for grp in error_op]
        return ops if group else ungroup(ops)

    @staticmethod
    def AD_full(n, Y=0.0):
        keys = ["A0", "A1"] * n
        return [GAD.A(int(x[-1]), 2, Y) for x in permut(keys, n)]

    @staticmethod
    def Pauli(n, order, p=0.0, paulis=["X", "Y", "Z"], group=False):
        keys = (["I"] + paulis) * n
        val = {"I": 0, "X": 1, "Y": 1, "Z": 1}
        error_op = [[] for _ in range(order + 1)]
        for comb in permut(keys, n):
            ord = sum(val[x.upper()] for x in comb)
            if ord <= order:
                error_op[ord].append(comb)

        def gen(x):
            return getattr(Pauli, x.upper())(p)

        ops = [[gen(x) for x in grp] for grp in error_op]
        return ops if group else ungroup(ops)

    @staticmethod
    def Pauli_full(n, p=0.0, paulis=["I", "X", "Y", "Z"]):
        keys = paulis * n
        return [getattr(Pauli, x.upper())(p) for x in permut(keys, n)]

    @staticmethod
    def GAD(n, order, keys=["A0", "A1", "R1"], Y=0.0, p=0.0, group=False):
        keys *= n
        error_op = [[] for _ in range((order + 1) * 2 - 1)]
        for comb in permut(keys, n):
            s = sum(int(x[-1]) for x in comb)
            if s <= order:
                idx = (
                    2 * s - 1
                    if any("R" in x and int(x[-1]) > 0 for x in comb)
                    else 2 * s
                )
                error_op[idx].append(comb)

        def gen(e):
            if "A" in e:
                return GAD.A(int(e[-1]), 2, Y)
            return GAD.R(int(e[-1]), 2, Y)

        ops = [[gen(x) for x in grp] for grp in error_op]
        return ops if group else ungroup(ops)

    @staticmethod
    def GAD_full(n, Y, p):
        # WHY IS THERE NO P USED HERE
        keys = ["A0", "A1", "R0", "R1"] * n

        def gen(e):
            if "A" in e:
                return GAD.A(int(e[-1]), 2, Y)
            return GAD.R(int(e[-1]), 2, Y)

        return [gen(x) for x in permut(keys, n)]
