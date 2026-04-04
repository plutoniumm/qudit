from MDR import Exam, load, Question
import sys
import torch as pt

sys.path.append("..")

from qudit.noise import Process
from qudit.qec import Recovery
from qudit.qec.lib import Leung


def to_rho(x):
    size = x.numel()
    return (x.view(size, 1) @ pt.conj(x.view(1, size))).to(pt.complex64)


fid = lambda rho, sigma: pt.real(pt.trace(rho @ sigma)).item()


class QEC(Question):
    """
    Quantum error-correction tests: code orthonormality and
    recovery map fidelity on noisy codewords.
    """

    def test_leung_orthonormal(self):
        """
        Leung codewords are orthonormal:
        $\\langle 0_L | 0_L \\rangle = \\langle 1_L | 1_L \\rangle = 1$,
        $\\langle 0_L | 1_L \\rangle = 0$
        """
        state0, state1 = Leung().toTensor()
        inner_00 = pt.dot(state0, state0).real.item()
        inner_11 = pt.dot(state1, state1).real.item()
        inner_01 = pt.abs(pt.dot(state0, state1)).item()
        self.assertAlmostEqual(inner_00, 1.0, places=6)
        self.assertAlmostEqual(inner_11, 1.0, places=6)
        self.assertAlmostEqual(inner_01, 0.0, places=6)

    def test_petz_ad(self):
        """
        $\\mathcal{R}_\\mathrm{Petz}$ on amplitude-damping noise ($\\gamma=0.1$,
        order 3) with Leung 4-qubit codewords recovers fidelity $\\approx 0.9889$
        and strictly improves on the noisy fidelity
        """
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
            fid0_clean,
            0.9889,
            places=2,
            msg="Petz AD recovery fidelity mismatch for codeword 0",
        )
        self.assertAlmostEqual(
            fid1_clean,
            0.9889,
            places=2,
            msg="Petz AD recovery fidelity mismatch for codeword 1",
        )
        self.assertGreater(
            fid0_clean,
            fid0_noisy,
            msg="Recovery should improve fidelity for codeword 0",
        )
        self.assertGreater(
            fid1_clean,
            fid1_noisy,
            msg="Recovery should improve fidelity for codeword 1",
        )


if __name__ == "__main__":
    runner = Exam(
        name="Qudit ECC Tests",
        desc="Validation of error-correction code properties and recovery maps",
        file="ECC.md",
    )
    runner.run(load(QEC))
