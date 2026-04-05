from .codes import Code
from .stab import Stabilisers
import torch as pt
import math


def Dutta3() -> Code:
    """
    Dutta's 3-qubit code, which is a permutation-invariant code designed to be the smallest code that can correct a single AD error. The codewords are:

    $
    |0_L\\rangle = \\frac{1}{\sqrt{3}}(|001\\rangle + |010\\rangle + |100\\rangle)$
    """

    cw0 = pt.zeros(8)
    cw1 = pt.zeros(8)

    cw0[1] = 1
    cw0[2] = 1
    cw0[4] = 1
    cw1[7] = 1

    code = pt.stack([cw0 / pt.linalg.norm(cw0), cw1 / pt.linalg.norm(cw1)])

    return Code(code)


def Leung() -> Code:
    """
    Leung's 4-qubit code, which is a standard code that can correct a single AD error. The codewords are:

    $ |0_L\\rangle = \\frac{1}{\sqrt{2}}(|0000\\rangle + |1111\\rangle) \\\\ |1_L\\rangle = \\frac{1}{\sqrt{2}}(|0011\\rangle + |1100\\rangle)
    """

    leung_0 = pt.zeros(16)
    leung_1 = pt.zeros(16)

    leung_0[0] = 1
    leung_0[-1] = 1
    leung_1[3] = 1
    leung_1[12] = 1

    code = pt.stack(
        [leung_0 / pt.linalg.norm(leung_0), leung_1 / pt.linalg.norm(leung_1)]
    )

    return Code(code)


def Perfect() -> Code:
    """
    The $[[5, 1, 3]]$ Perfect code, which is the smallest code that can correct an arbitrary single-qubit error. The codewords are:

    $|0_L\\rangle = \\frac{1}{4}(|00000\\rangle + |11000\\rangle + |01100\\rangle + |00110\\rangle + |00011\\rangle - |10001\\rangle - |01001\\rangle - |00101\\rangle - |00010\\rangle - |10000\\rangle - |01000\\rangle - |00100\\rangle - |11110\\rangle - |11101\\rangle - |11011\\rangle - |10111\\rangle)$

    $|1_L\\rangle = X^{\\otimes 5}|0_L\\rangle$
    """

    _0L = pt.zeros(32)
    _1L = pt.zeros(32)

    k0 = [0, 18, 9, 20, 10, -27, -6, -24, -29, -3, -30, -15, -17, -12, -23, 5]
    k1 = [31, 13, 22, 11, 21, -4, -25, -7, -2, -28, -1, -16, -14, -19, -8, 26]

    for i in range(len(k0)):
        _0key = k0[i]
        _1key = k1[i]

        _0L[abs(_0key)] = 1.0 if _0key >= 0 else -1.0
        _1L[abs(_1key)] = 1.0 if _1key >= 0 else -1.0

    _0L[0] = 1

    return Code(pt.stack([_0L / pt.linalg.norm(_0L), _1L / pt.linalg.norm(_1L)]))


def Qutrit3() -> Code:
    """
    3-qutrit repetition code: three codewords $|0_L\\rangle=|000\\rangle$,
    $|1_L\\rangle=|111\\rangle$, $|2_L\\rangle=|222\\rangle$.

    Encodes one logical qutrit in three physical qutrits ($d^n=3^3=27$ dimensional space).
    Detects (and with measurement corrects) single qutrit shift ($X_3$) errors.
    """
    cw = pt.zeros(3, 27)
    cw[0, 0] = 1.0
    cw[1, 13] = 1.0
    cw[2, 26] = 1.0

    return Code(cw, d=3)


def GottesmanD(d: int = 2) -> Code:
    """
    Gottesman-type CSS code for prime $d$: encodes one logical $d$-level qudit in $d$
    physical qudits.

    Codewords: $|j_L\\rangle = \\frac{1}{\\sqrt{d}}\\sum_{a=0}^{d-1}|a,\\,a{+}j,\\,a{+}2j,\\,\\ldots\\rangle \\pmod{d}$

    For $d=2$: $|0_L\\rangle=(|00\\rangle+|11\\rangle)/\\sqrt{2}$, $|1_L\\rangle=(|01\\rangle+|10\\rangle)/\\sqrt{2}$.
    For $d=3$: three orthonormal codewords on 3 physical qutrits (27-dimensional space).
    """
    n_phys = d
    dim_total = d**n_phys
    cws = []
    norm = 1.0 / math.sqrt(d)
    for j in range(d):
        vec = pt.zeros(dim_total, dtype=pt.complex64)
        for a in range(d):
            idx = 0
            for pos in range(n_phys):
                digit = (a + pos * j) % d
                idx = idx * d + digit
            vec[idx] += norm
        cws.append(vec)

    return Code(pt.stack(cws), d=d)


def Surface(m: int, n: int, d: int = 2, edge: str = "even", start: str = "X") -> Code:
    """
    Build an $m \\times n$ surface code and return a ``Code`` spanning the stabilizer space.
    """
    stabs = Stabilisers(m, n, edge=edge, start=start)

    return Code.fromStabilizers(stabs, d=d)
