import sys

sys.path.append("..")
from qudit import Basis, State, Unity
from unittest import TestCase, main
import numpy as np


class States(TestCase):
    def test_basis(self):
        Ket = Basis(2)

        K00 = Ket("00")

        self.assertTrue(np.allclose(Ket(0, 0), K00))
        self.assertTrue(np.allclose(Ket(0) ^ Ket(0), K00))

        self.assertTrue(np.sum(K00) == 1)
        self.assertTrue(np.abs(K00[0]) == 1)

    def test_state_construction(self):
        pi = np.pi
        e, rt = np.exp, np.sqrt
        cos, sin = np.cos, np.sin

        w = Unity(3)
        Ket = Basis(4)
        SV = State(
            w * Ket("0000")
            + w**2 * Ket("1010")
            + rt(3) * 1j * Ket("2010")
            + Ket("2200")
            + (9j + 16) * Ket("1210")
            + (w - w**2) * Ket("0022")
            + (w - 1) ** 2 * Ket("2020")
            + (e(1j * pi / 18) + 6) * Ket("2221")
            + Ket("0112")
            + (5 + 9j) * Ket("1200")
            + 0.67 * Ket("1111")
            + (9 * cos(pi / 16) + 1j * sin(pi / 5)) * Ket("2222")
        )

        self.assertEqual(SV.shape, (4**4,))
        self.assertTrue(np.iscomplexobj(SV))

        self.assertAlmostEqual(SV[0], 0.0208 + 3.597e-02j, places=3)

        self.assertAlmostEqual(SV[0], 0.0208 + 3.597e-02j, places=3)


if __name__ == "__main__":
    main()
