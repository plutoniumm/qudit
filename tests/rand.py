import sys

sys.path.append("..")
from unittest import TestCase, main
from qudit.random import random_unitary, random_state
from unittest import TestCase, main
import numpy as np


class Random(TestCase):
    def test_unitary_mean(self):
        U = rand.unitary(20)
        all = np.concatenate((U.real, U.imag))
        self.assertLess(abs(np.mean(all)), 0.05)

    def test_state_mean(self):
        S = rand.state(20)
        all = np.concatenate((S.real, S.imag))
        self.assertLess(abs(np.mean(all)), 0.05)


if __name__ == "__main__":
    main()
    main()
