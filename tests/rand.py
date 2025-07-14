import sys

sys.path.append("..")
from qudit.random import random_unitary, random_state
from unittest import TestCase, main
import numpy as np


class Random(TestCase):
    def test_unitary_mean(self):
        U = random_unitary(20)
        all = np.concatenate((U.real, U.imag))
        self.assertLess(abs(np.mean(all)), 0.05)

    def test_state_mean(self):
        S = random_state(20)
        all = np.concatenate((S.real, S.imag))
        self.assertLess(abs(np.mean(all)), 0.05)


if __name__ == "__main__":
    main()
