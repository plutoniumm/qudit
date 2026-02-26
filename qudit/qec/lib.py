from .codes import Code
import torch as pt
import numpy as np


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

        _0L[np.abs(_0key)] = np.sign(_0key)
        _1L[np.abs(_1key)] = np.sign(_1key)

    _0L[0] = 1

    return Code(pt.stack([_0L, _1L]))
