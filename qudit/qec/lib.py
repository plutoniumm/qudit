from .codes import Code
import torch as pt
import math
import sys
import os


def Dutta3() -> Code:
    """
    Dutta's 3-qubit code, which is a permutation-invariant code designed to be the smallest code that can correct a single AD error. The codewords are:

    $
    |0_L\\rangle = \\frac{1}{\sqrt{3}}(|001\\rangle + |010\\rangle + |100\\rangle)$
    """

    dutta_3_0 = pt.zeros(8)
    dutta_3_1 = pt.zeros(8)

    dutta_3_0[1] = 1
    dutta_3_0[2] = 1
    dutta_3_0[4] = 1
    dutta_3_1[7] = 1

    code = pt.stack(
        [dutta_3_0 / pt.linalg.norm(dutta_3_0), dutta_3_1 / pt.linalg.norm(dutta_3_1)]
    )

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

    _0L_keys = [0, 18, 9, 20, 10, -27, -6, -24, -29, -3, -30, -15, -17, -12, -23, 5]
    _1L_keys = [31, 13, 22, 11, 21, -4, -25, -7, -2, -28, -1, -16, -14, -19, -8, 26]

    for i in range(len(_0L_keys)):
        _0key = _0L_keys[i]
        _1key = _1L_keys[i]

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
    cw[0, 0] = 1.0  # |000> = index 0
    cw[1, 13] = 1.0  # |111> = 1*9 + 1*3 + 1 = 13
    cw[2, 26] = 1.0  # |222> = 2*9 + 2*3 + 2 = 26

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
    Build an $m \times n$ surface code and return a ``Code`` spanning the stabilizer space.

    Requires the ``surface/`` directory (containing ``stab.py``) at the repository root.
    """
    _surface = os.path.join(os.path.dirname(__file__), "../../surface")
    if _surface not in sys.path:
        sys.path.insert(0, _surface)
    from stab import Stabilisers

    stabs = Stabilisers(m, n, edge=edge, start=start)

    return Code.fromStabilizers(stabs, d=d)
