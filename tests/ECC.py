from MDR import Exam, load, Question
import sys
import numpy as np
from numpy import linalg as LA

sys.path.append("..")
from qudit.noise import Process, Channel
from qudit.tools import Fidelity
from qudit.qec import Recovery


class QEC(Question):
    """
    Error-correction tests using the MDR framework.
    """

    code = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0.0],
        ],
        dtype=np.complex64,
    )
    code /= LA.norm(code, axis=1)[:, None]

    ops = Process.GAD(2, 4, Y=0.01, p=0.001)
    kraus = None

    def test_petz_recovery(self):
        rec = Recovery.petz(self.ops, self.code)
        fid = Fidelity.entanglement(rec, self.ops, self.code)

        self.assertAlmostEqual(
            fid, 0.98, places=2, msg="Petz recovery fidelity mismatch"
        )

    def test_leung_recovery(self):
        Ek = self.ops.correctable()

        rec = Recovery.leung(Ek, self.code)
        fid = Fidelity.entanglement(rec, self.ops, self.code)

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
