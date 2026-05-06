from MDR import Exam, load, Question
import sys
import numpy as np
import torch as pt

sys.path.append("..")

from qudit.noise import Process


class ChannelAnalysisTests(Question):
    """
    Channel.isTP, isCP, isCPTP, toChoi, toSuperop, toStinespring.
    """

    def test_ad_low_noise_is_cptp(self):
        """
        Amplitude damping with Y=0.01 (near-identity) is CPTP
        """
        ch = Process.AD(d=2, n=1, Y=0.01)

        self.assertTrue(ch.isTP, msg="AD(Y=0.01) should be trace-preserving")

        self.assertTrue(ch.isCP, msg="AD(Y=0.01) should be completely positive")

    def test_ad_noisy_is_cptp(self):
        """
        Amplitude damping with Y=0.3 is CPTP
        """
        ch = Process.AD(d=2, n=1, Y=0.3)

        self.assertTrue(ch.isCPTP, msg="AD(Y=0.3) should be CPTP")

    def test_gad_is_cptp(self):
        """
        GAD channel (Y=0.2, p=0.5) is CPTP
        """
        ch = Process.GAD(d=2, n=1, Y=0.2, p=0.5)

        self.assertTrue(ch.isCPTP, msg="GAD channel should be CPTP")

    def test_pauli_is_cptp(self):
        """
        Pauli channel with p=[0.1,0.1,0.1] is CPTP
        """
        ch = Process.Pauli(n=1, p=[0.1, 0.1, 0.1])

        self.assertTrue(ch.isCPTP, msg="Pauli channel should be CPTP")

    def test_choi_psd(self):
        """
        Choi matrix of a CPTP channel is positive semidefinite
        """
        ch = Process.AD(d=2, n=1, Y=0.2)
        J = ch.toChoi()
        eig = pt.linalg.eigvalsh(J.real.float())

        self.assertTrue(
            bool((eig >= -1e-5).all()),
            msg="Choi matrix should be positive semidefinite",
        )

    def test_choi_trace(self):
        """
        Choi matrix of a d=2 channel has trace = d = 2
        """
        ch = Process.AD(d=2, n=1, Y=0.1)
        J = ch.toChoi()
        trace_val = float(pt.trace(J.real).item())

        self.assertAlmostEqual(
            trace_val, 2.0, places=4, msg="Choi matrix trace should equal d=2"
        )

    def test_superop_shape(self):
        """
        Superoperator of a single-qubit channel is 4×4
        """
        ch = Process.Pauli(n=1, p=[0.1, 0.0, 0.0])
        S = ch.toSuperop()

        self.assertEqual(
            tuple(S.shape), (4, 4), msg="Single-qubit superoperator should be 4×4"
        )

    def test_stinespring_isometry(self):
        """
        Stinespring V satisfies V†V = I
        """
        ch = Process.AD(d=2, n=1, Y=0.2)
        V = ch.toStinespring()
        VdV = V.conj().T @ V
        I = pt.eye(VdV.shape[0], dtype=VdV.dtype)

        self.assertTrue(
            bool(pt.allclose(VdV, I, atol=1e-5)),
            msg="Stinespring V should satisfy V†V = I",
        )

    def test_ad_qutrit_is_cptp(self):
        """
        Qutrit amplitude damping (d=3) is CPTP
        """
        ch = Process.AD(d=3, n=1, Y=0.2)

        self.assertTrue(ch.isCPTP, msg="Qutrit AD channel should be CPTP")

    def test_pauli_2qubit_is_tp(self):
        """
        2-qubit Pauli channel is trace-preserving
        """
        ch = Process.Pauli(n=2, p=[0.05, 0.0, 0.0])

        self.assertTrue(ch.isTP, msg="2-qubit Pauli channel should be trace-preserving")


class WeylChannelTests(Question):
    """
    WeylNoise Kraus stack: trace-preserving for d=2, d=3, and identity at zero noise.
    """

    def test_weyl_d2_is_tp(self):
        """
        Weyl channel on a qubit ($d=2$) is trace-preserving: $\\sum_k K_k^\\dagger K_k = I$
        """
        from qudit.noise import WeylNoise

        K = WeylNoise(p=0.1).kraus_for("H", 0, 2).to(pt.complex64)
        total = sum(k.conj().T @ k for k in K)

        self.assertEqual(
            len(K), 4, msg="Weyl qubit channel should have 4 Kraus operators"
        )

        self.assertTrue(
            pt.allclose(total.real, pt.eye(2), atol=1e-5),
            msg="Weyl(d=2) should satisfy ∑K†K=I",
        )

    def test_weyl_d3_is_tp(self):
        """
        Weyl channel on a qutrit ($d=3$) is trace-preserving: $\\sum_k K_k^\\dagger K_k = I$
        """
        from qudit.noise import WeylNoise

        K = WeylNoise(p=0.02).kraus_for("H", 0, 3).to(pt.complex64)
        total = sum(k.conj().T @ k for k in K)

        self.assertEqual(
            len(K), 9, msg="Weyl qutrit channel should have 9 Kraus operators"
        )

        self.assertTrue(
            pt.allclose(total.real, pt.eye(3), atol=1e-5),
            msg="Weyl(d=3) should satisfy ∑K†K=I",
        )

    def test_weyl_identity_at_zero_noise(self):
        """
        Weyl channel with p=0 is the identity channel: $K_0 = I$, all others zero
        """
        from qudit.noise import WeylNoise

        K = WeylNoise(p=0.0).kraus_for("H", 0, 2).to(pt.complex64)

        self.assertEqual(
            len(K),
            4,
            msg="Zero-noise Weyl channel should still have 4 Kraus operators",
        )

        self.assertTrue(
            pt.allclose(K[0].real, pt.eye(2), atol=1e-5),
            msg="K_0 should be identity at zero noise",
        )

        for k in K[1:]:
            self.assertTrue(
                pt.allclose(k.abs(), pt.zeros(2, 2), atol=1e-5),
                msg="Non-identity Kraus ops should be zero at zero noise",
            )


if __name__ == "__main__":
    from MDR import Exam, load

    runner = Exam(
        name="Channel Analysis Tests",
        desc="isTP, isCP, isCPTP, toChoi, toSuperop, toStinespring, Weyl",
        file="channel.md",
    )
    runner.run(load(ChannelAnalysisTests))
    runner.run(load(WeylChannelTests))
