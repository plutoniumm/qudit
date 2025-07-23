import sys
import unittest
import numpy as np
from numpy import linalg as LA

sys.path.append("..")
from unittest import TestCase, main
from qudit.noise import Recovery, Process
from qudit.tools import Fidelity


class QEC(unittest.TestCase):
    def setUp(self):
        self.code = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0.0],
            ],
            dtype=np.complex64,
        )
        self.code /= LA.norm(self.code, axis=1)[:, None]
        self.ops = Process.GAD(2, 4, Y=0.01, p=0.001)

    def test_petz_recovery(self):
        rec = Recovery.petz(self.ops, self.code)
        fid = Fidelity.entanglement(rec, self.ops, self.code)

        self.assertAlmostEqual(fid, 0.98, places=2)

    def test_leung_recovery(self):
        Ek = self.ops.correctable()

        rec = Recovery.leung(Ek, self.code)
        fid = Fidelity.entanglement(rec, self.ops, self.code)

        self.assertAlmostEqual(fid, 0.91, places=2)


if __name__ == "__main__":
    unittest.main()
