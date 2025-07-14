from numpy.random import normal as N
import numpy.linalg as LA
import numpy as np

C128 = np.complex128

"""
Potential references:
- [Composite parameterization and Haar measure for
all unitary and special unitary groups](https://arxiv.org/pdf/1103.3408) - Explict expression for haar measure.
"""


# SRC: https://case.edu/artsci/math/mwmeckes/elizabeth/Meckes_SAMSI_Lecture2.pdf
def random_unitary(n: int) -> np.ndarray:
    l, r = N(size=(n, n)).astype(C128), N(size=(n, n)).astype(C128)
    Q, R = LA.qr(l + 1j * r)

    # Rii / |Rii|
    A = np.diag([R[i, i] / np.abs(R[i, i]) for i in range(n)])

    return np.dot(Q, A)


def random_state(n: int) -> np.ndarray:
    U = random_unitary(n)
    vec = np.eye(n, dtype=C128)
    vec = vec[np.random.randint(0, n)]

    vec = np.dot(U, vec)
    vec /= LA.norm(vec)
    return vec.astype(C128)
