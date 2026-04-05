from scipy.special import factorial as F, gammainc as g
from itertools import combinations
from ..index import State, Basis
import numpy as np


def GHZ(n: int, d: int) -> State:
    """
    Construct an $n$-partite qudit GHZ state.

    Returns $\sum_{i=0}^{d-1} |i\\rangle^{\otimes n}$ (unnormalized).
    """
    Ket = Basis(d)
    vals = sum([Ket(f"{i}" * n) for i in range(d)])

    return State(vals)


def W(n: int, d: int = 2) -> State:
    """
    Construct the $n$-qudit $W$ state (single-excitation symmetric state).

    Returns $\sum_{i=0}^{n-1} |0\cdots 1_i \cdots 0\\rangle$ (unnormalized)
    where $|0\\rangle$ and $|1\\rangle$ are the ground and first-excited levels.
    """
    Ket = Basis(d)
    vals = ["0" * i + "1" + "0" * (n - i - 1) for i in range(n)]
    vals = sum([Ket(v) for v in vals])

    return State(vals)


def NOON(n: int, theta: float = 0.0) -> State:
    """
    Construct a two-mode NOON-like state.

    Returns $|n,0\\rangle + e^{i n\\theta}|0,n\\rangle$ (unnormalized) in a local dimension $n+1$.
    """
    Ket = Basis(n + 1)
    kets = [Ket(0, n), Ket(n, 0)]

    kets = kets[0] + np.exp(1j * n * theta) * kets[1]

    return State(kets)


def Dicke(n: int, k: int, d: int = 2) -> State:
    """
    Construct an $n$-qudit Dicke state with $k$ zeros (and $n-k$ ones).

    Returns the uniform (unnormalized) superposition over all distinct permutations of
    $k$ symbols $0$ and $n-k$ symbols $1$.
    """
    Ket = Basis(d)
    vals = sum(
        Ket("".join("0" if i in zeros else "1" for i in range(n)))
        for zeros in combinations(range(n), k)
    )

    return State(vals)


def Coherent(N: int, alpha: complex = 1.0) -> State:
    """
    Truncated coherent state in a finite $(N+1)$-dimensional Fock space.

    Builds $\sum_{n=0}^{N} \\alpha^n/\sqrt{n!}\,|n\\rangle$ with a normalization correction from
    the incomplete gamma function.

    [Reference at arxiv:1402.1487](https://arxiv.org/pdf/1402.1487)
    """
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
