from MDR import Exam, load, Question
import sys
import numpy as np
from numpy import linalg as LA

sys.path.append("..")
from qudit.noise import Recovery, Process, Channel
from qudit.tools import Fidelity


class QEC(Question):
    """
    Error-correction tests using the MDR framework.
    """

    code = None
    ops: Channel
    kraus = None
    codes = None

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
        self.kraus = self.ops.kraus()
        self.codes = list(self.code)

    def test_petz_recovery(self):
        rec = Recovery.petz(self.kraus, self.codes)
        fid = Fidelity.entanglement(rec, self.kraus, self.codes)
        self.assertAlmostEqual(
            fid, 0.98, places=2, msg="Petz recovery fidelity mismatch"
        )

    def test_leung_recovery(self):
        Ek = self.ops.correctable()
        rec = Recovery.leung(Ek, self.codes)
        fid = Fidelity.entanglement(rec, self.kraus, self.codes)
        self.assertAlmostEqual(
            fid, 0.91, places=2, msg="Leung recovery fidelity mismatch"
        )


if __name__ == "__main__":
    runner = Exam(
        name="Qudit ECC Tests",
        desc="Validation of error-correction recoveries",
        file="ECC.md",
    )
    runner.run(load(QEC))
