from scipy.special import factorial as F, gammainc as g
from itertools import permutations
from ..index import State, Basis
import numpy as np


def GHZ(n: int, d: int) -> State:
    # sum_ i^d -> |0000> + |1111> + |2222> for n=4, d=3
    Ket = Basis(d)
    vals = sum([Ket(f"{i}" * n) for i in range(d)])

    return State(vals)


def W(n: int) -> State:
    # |100> + |010> + |001> for n=3
    Ket = Basis(2)
    vals = ["0" * i + "1" + "0" * (n - i - 1) for i in range(n)]
    vals = sum([Ket(v) for v in vals])
    return State(vals)


def NOON(n: int, theta: float = 0) -> State:
    # |N0> + |0N> for N
    Ket = Basis(n + 1)
    kets = [Ket(0, n), Ket(n, 0)]

    kets = kets[0] + np.exp(1j * n * theta) * kets[1]
    return State(kets)


def Dicke(n, k):
    # |1000> + |0100> + |0010> + |0001> for n=4, k=1
    Ket = Basis(2)
    vals = ["0"] * k + ["1"] * (n - k)
    vals = set(["".join(p) for p in permutations(vals)])
    vals = sum([Ket(v) for v in vals])

    return State(vals)


# SRC: https://arxiv.org/pdf/1402.1487
def Coherent(N: int, alpha=1.0) -> State:
    Ket = Basis(N + 1)
    a2 = abs(alpha) ** 2

    norm = np.exp(a2) * (1 - g(N + 1, a2) / F(N))
    norm = 1 / np.sqrt(norm)

    vec = sum((alpha**n / np.sqrt(F(n))) * Ket(n) for n in range(N + 1))
    return State(norm * vec)


if __name__ == "__main__":
    print(GHZ(2, 2))
    print(W(3))
    print(NOON(3, np.pi / 4))
    print(Dicke(4, 1))
    print(Coherent(5, alpha=1.0))
