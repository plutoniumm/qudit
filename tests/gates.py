import sys
import numpy as np
from unittest import TestCase, main

sys.path.append("..")
from qudit import Gategen


class Gates(TestCase):
    def setUp(self):
        self.D = Gategen(2)

    def matEqual(self, A, B, places=6):
        self.assertEqual(A.shape, B.shape)
        np.testing.assert_almost_equal(A, B, decimal=places)

    def test_X(self):
        X = np.array([[0, 1], [1, 0]])
        self.matEqual(X, self.D.X)

    def test_Z(self):
        Z = np.array([[1, 0], [0, -1]])
        self.matEqual(Z, self.D.Z)

    def test_H(self):
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        self.matEqual(H, self.D.H)


if __name__ == "__main__":
    main()
