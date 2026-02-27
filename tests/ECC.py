from MDR import Exam, load, Question
import sys
import torch as pt

sys.path.append("..")
from qudit.noise import Process, Channel
from qudit.qec import Recovery
from qudit.qec.lib import Leung

def to_rho(x):
    size = x.numel()
    return (x.view(size, 1) @ pt.conj(x.view(1, size))).to(pt.complex64)

fid = lambda rho, sigma: pt.real(pt.trace(rho @ sigma)).item()

class QEC(Question):
    """
    Error-correction tests using the MDR framework.
    """

    def test_petz_ad(self):
        """Test Petz recovery on AD noise using Leung codewords (from testnoise0.py)."""

        n = 4
        state0, state1 = Leung().toTensor()
        rho0, rho1 = to_rho(state0), to_rho(state1)

        noise = Process.AD(d=2, n=n, Y=0.1, order=3)

        noisy0 = noise.run(rho0)
        noisy1 = noise.run(rho1)

        rec = Recovery.petz(noise, [state0, state1])

        clean0 = rec.run(noisy0)
        clean1 = rec.run(noisy1)

        fid0_noisy = fid(rho0, noisy0)
        fid0_clean = fid(rho0, clean0)

        fid1_noisy = fid(rho1, noisy1)
        fid1_clean = fid(rho1, clean1)

        self.assertAlmostEqual(
            fid0_clean, 0.9889, places=2, msg="Petz AD recovery fidelity mismatch for codeword 0"
        )
        self.assertAlmostEqual(
            fid1_clean, 0.9889, places=2, msg="Petz AD recovery fidelity mismatch for codeword 1"
        )
        self.assertGreater(
            fid0_clean, fid0_noisy, msg="Recovery should improve fidelity for codeword 0"
        )
        self.assertGreater(
            fid1_clean, fid1_noisy, msg="Recovery should improve fidelity for codeword 1"
        )


if __name__ == "__main__":
    runner = Exam(
        name="Qudit ECC Tests",
        desc="Validation of error-correction recoveries",
        file="ECC.md",
    )
    runner.run(load(QEC))
