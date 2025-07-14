import sys

sys.path.append("..")
from qudit.tools.entanglement import Loss, rank, Perp
from unittest import TestCase, main
from qudit import Basis, State
import numpy as np

THETA, D, r = 1.5, 5, 2
Bits, Trits = Basis(2), Basis(3)


def Psi(i):
    A = Bits(0) ^ Trits(i)
    B = Bits(1) ^ Trits(i + 1)
    return A * np.cos(THETA / 2) + B * np.sin(THETA / 2)


class Ranken(TestCase):
    def system(self, X):
        qbit = State(X[1:5].reshape(2, 2).dot([1, 1j]))
        qtrit = State(X[5:11].reshape(3, 2).dot([1, 1j]))
        phi_rx = (X[0] * (qbit ^ qtrit)).norm()

        return Loss(phi_rx, self.perp)

    def test_rank(self):
        self.perp = Perp([Psi(i) for i in range(2)])

        res = rank(self.system, D, r, tries=2)
        self.assertIsInstance(res, float)
        self.assertGreater(res, 0)
        self.assertTrue(res - 0.2481 < 1e-4, "Expected value close to 0.2481")


if __name__ == "__main__":
    main()
