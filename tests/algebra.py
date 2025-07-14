import sys

sys.path.append("..")
from unittest import TestCase, main
from qudit import dGellMann


class GellMann(TestCase):
    n = 3

    def test_n(self):
        gm = dGellMann(self.n)
        # I return identity also so (n^2 - 1) + 1
        self.assertEqual(len(gm), self.n**2)

    def test_shape(self):
        gm = dGellMann(self.n)

        for mat in gm:
            self.assertTrue(hasattr(mat, "shape"))
            self.assertEqual(mat.shape, (self.n, self.n))


if __name__ == "__main__":
    main()
