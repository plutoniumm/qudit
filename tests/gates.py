import sys
import numpy as np
from unittest import TestCase, main

sys.path.append("..")
from qudit import Gategen, tensorise
import torch


class Gates(TestCase):
    def setUp(self):
        self.D = Gategen(2)

    def matEqual(self, A, B):
        self.assertEqual(A.shape, B.shape)
        A, B = tensorise(A), tensorise(B)

        diff = round(torch.sum(torch.abs(A - B)).item(), 4)
        self.assertEqual(diff, 0)

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
