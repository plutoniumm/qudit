import numpy as np
import torch as pt

C128 = pt.complex128

"""
Potential references:
- [Composite parameterization and Haar measure for
all unitary and special unitary groups](https://arxiv.org/pdf/1103.3408) - Explict expression for Haar measure.
"""


# SRC: https://case.edu/artsci/math/mwmeckes/elizabeth/Meckes_SAMSI_Lecture2.pdf
def random_unitary(n: int) -> pt.Tensor:
    """
    Sample a Haar-random unitary $U\in U(n)$.

    Constructs a complex Ginibre matrix $Z$ and returns the $Q$ factor of $Z=QR$ with diagonal phase correction
    $Q \mapsto Q\,\mathrm{diag}(R_{ii}/|R_{ii}|)$.
    """
    l = pt.from_numpy(np.random.randn(n, n))
    r = pt.from_numpy(np.random.randn(n, n))
    Z = (l + 1j * r).to(C128)
    Q, R = pt.linalg.qr(Z)

    # Phase correction: Rii / |Rii|
    phases = pt.tensor([R[i, i] / pt.abs(R[i, i]) for i in range(n)], dtype=C128)
    A = pt.diag(phases)

    return Q @ A


def random_state(n: int) -> pt.Tensor:
    """
    Sample a Haar-random pure state $|\psi\\rangle \in \mathbb{C}^n$.

    Draws a Haar unitary $U$ and applies it to a uniformly chosen computational basis vector, then normalizes.
    """
    U = random_unitary(n)
    vec = pt.eye(n, dtype=C128)
    vec = vec[int(np.random.randint(0, n))]

    vec = U @ vec
    vec = vec / pt.linalg.norm(vec)

    return vec.to(C128)
