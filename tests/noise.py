from MDR import Exam, load, Question
import sys

sys.path.append("..")

from qudit.noise.lib import Process, IID
from qudit.noise.index import Channel, Multiplex
import numpy as np
import torch as pt

C64 = pt.complex64


def rho0_qubit():
    """
    Density matrix |0><0| for a single qubit.
    """
    r = pt.zeros((2, 2), dtype=C64)
    r[0, 0] = 1.0

    return r


def rho_mixed_qubit():
    """
    Maximally mixed qubit I/2.
    """
    return (0.5 * pt.eye(2, dtype=C64)).to(C64)


class ChannelProperties(Question):
    """
    Channel.isTP, isCP, isCPTP, toChoi, toSuperop, toStinespring tests
    using amplitude damping as a known-CPTP example.
    """

    def _ad_channel(self, n=1, Y=0.1, order=2):
        return Process.AD(d=2, n=n, Y=Y, order=order)

    def _gad_channel(self, n=1, Y=0.1, p=0.5):
        return Process.GAD(d=2, n=n, Y=Y, p=p, order=1)

    def test_ad_is_tp(self):
        """

        $\\sum_k E_k^\\dagger E_k = I$: amplitude damping channel ($Y=0.1$) is trace-preserving
        """

        ch = self._ad_channel()

        self.assertTrue(ch.isTP, "AD channel should be trace-preserving")

    def test_ad_is_cp(self):
        """

        Choi matrix $\\succeq 0$: amplitude damping channel ($Y=0.1$) is completely positive
        """

        ch = self._ad_channel()

        self.assertTrue(ch.isCP, "AD channel should be completely positive")

    def test_ad_is_cptp(self):
        """

        Amplitude damping channel ($Y=0.1$, order 2) is CPTP
        """

        ch = self._ad_channel()

        self.assertTrue(ch.isCPTP, "AD channel should be CPTP")

    def test_gad_is_tp(self):
        """

        $\\sum_k E_k^\\dagger E_k = I$: generalized amplitude damping ($Y=0.1$, $p=0.5$) is TP
        """

        ch = self._gad_channel()

        self.assertTrue(ch.isTP, "GAD channel should be trace-preserving")

    def test_gad_is_cptp(self):
        """

        Generalized amplitude damping ($Y=0.1$, $p=0.5$) is CPTP
        """

        ch = self._gad_channel()

        self.assertTrue(ch.isCPTP, "GAD channel should be CPTP")

    def test_choi_trace(self):
        """
        $\\mathrm{Tr}(J(\\Phi)) = d$: Choi matrix of a 1-qubit CPTP AD channel has trace $d=2$
        """

        ch = self._ad_channel()
        J = ch.toChoi()

        tr = float(pt.trace(J).real.item())

        self.assertAlmostEqual(
            tr, 2.0, places=4, msg="Choi matrix trace should equal d=2"
        )

    def test_choi_psd(self):
        """
        $J(\\Phi) \\succeq 0$: Choi matrix of CPTP channel is positive semi-definite
        """

        ch = self._ad_channel()
        J = ch.toChoi()

        evals = pt.linalg.eigvalsh(J.real.to(pt.float64))

        self.assertTrue(
            bool((evals >= -1e-6).all().item()),
            "Choi matrix of CPTP channel should be PSD",
        )

    def test_choi_hermitian(self):
        """
        $J(\\Phi)^\\dagger = J(\\Phi)$: Choi matrix is Hermitian
        """

        ch = self._ad_channel()
        J = ch.toChoi()

        diff = float(pt.norm(J - J.conj().T).real.item())

        self.assertAlmostEqual(
            diff, 0.0, places=4, msg="Choi matrix should be Hermitian"
        )

    def test_superop_shape(self):
        """
        $S \\in \\mathbb{C}^{d^2 \\times d^2}$: superoperator has correct shape $(4,4)$
        for a 1-qubit AD channel

        """
        ch = self._ad_channel()

        S = ch.toSuperop()

        self.assertEqual(
            tuple(S.shape),
            (4, 4),
            msg="Superoperator shape should be (d^2, d^2) = (4,4)",
        )

    def test_superop_trace_preserving(self):
        """
        TP channel: $S\\,\\mathrm{vec}(\\rho)$ has trace 1 when $\\rho$ has trace 1.
        Verified by checking $S\\,\\mathrm{vec}(|0\\rangle\\langle 0|)$ has trace 1.
        """
        ch = self._ad_channel()
        S = ch.toSuperop()
        d = 2
        # vec(|0><0|) = [1, 0, 0, 0]
        rho_vec = pt.tensor([1.0, 0.0, 0.0, 0.0], dtype=pt.complex128)

        out_vec = S @ rho_vec
        out_mat = out_vec.reshape(d, d)

        tr = float(pt.trace(out_mat).real.item())

        self.assertAlmostEqual(
            tr, 1.0, places=4, msg="Superoperator should preserve trace"
        )

    def test_stinespring_isometry(self):
        """
        $V^\\dagger V = I_d$: Stinespring isometry satisfies the isometry condition for AD channel
        """
        ch = self._ad_channel()
        V = ch.toStinespring()
        VdV = V.conj().T @ V

        d = 2
        identity = pt.eye(d, dtype=pt.complex128)

        diff = float(pt.norm(VdV - identity).real.item())

        self.assertAlmostEqual(diff, 0.0, places=4, msg="Stinespring isometry V†V ≠ I")

    def test_stinespring_shape(self):
        """
        $V \\in \\mathbb{C}^{(dr) \\times d}$: Stinespring matrix has correct shape
        """
        ch = self._ad_channel()
        V = ch.toStinespring()
        r = len(ch.ops)


        d = 2

        self.assertEqual(V.shape[1], d, "Stinespring matrix wrong column count")

        self.assertEqual(V.shape[0], d * r, "Stinespring matrix wrong row count")

    def test_ad_run_trace_preserved(self):
        """
        $\\mathrm{Tr}(\\Phi_{AD}(\\rho)) = 1$: AD channel preserves trace of $|0\\rangle\\langle0|$
        """
        ch = self._ad_channel()

        rho = rho0_qubit()
        out = ch.run(rho)

        tr = float(pt.trace(out).real.item())

        self.assertAlmostEqual(
            tr, 1.0, places=4, msg="AD channel should preserve trace"
        )

    def test_gad_run_trace_preserved(self):
        """
        $\\mathrm{Tr}(\\Phi_{GAD}(\\rho)) = 1$: GAD channel preserves trace
        """
        ch = self._gad_channel()

        rho = rho0_qubit()
        out = ch.run(rho)

        tr = float(pt.trace(out).real.item())

        self.assertAlmostEqual(
            tr, 1.0, places=4, msg="GAD channel should preserve trace"
        )

    def test_ad_dampens_excited(self):
        """
        AD channel reduces $|1\\rangle\\langle1|$ population:
        $\\langle 1|\\Phi_{AD}(|1\\rangle\\langle1|)|1\\rangle < 1$
        """
        ch = self._ad_channel(Y=0.5)
        rho1 = pt.zeros((2, 2), dtype=C64)

        rho1[1, 1] = 1.0
        out = ch.run(rho1)

        pop_excited = float(out[1, 1].real.item())

        self.assertLess(
            pop_excited, 1.0, msg="AD channel should damp excited population"
        )


class ProcessChannels(Question):
    """
    Process.AD, Process.GAD, IID.AD, IID.GAD channel construction and CPTP tests.
    """

    def test_ad_2qubit_tp(self):
        """

        $Process.AD(d=2, n=2, Y=0.1)$ is trace-preserving
        """

        ch = Process.AD(d=2, n=2, Y=0.1, order=2)

        self.assertTrue(ch.isTP, "2-qubit AD channel should be TP")

    def test_ad_trace_preserved_2q(self):
        """
        $\\mathrm{Tr}(\\Phi_{AD}(|00\\rangle\\langle00|)) = 1$: 2-qubit AD channel preserves trace
        """
        ch = Process.AD(d=2, n=2, Y=0.1, order=2)
        rho = pt.zeros((4, 4), dtype=C64)

        rho[0, 0] = 1.0
        out = ch.run(rho)

        tr = float(pt.trace(out).real.item())

        self.assertAlmostEqual(
            tr, 1.0, places=4, msg="2-qubit AD channel should preserve trace"
        )

    def test_gad_2qubit_tp(self):
        """

        $Process.GAD(d=2, n=2, Y=0.1, p=0.5)$ is trace-preserving
        """

        ch = Process.GAD(d=2, n=2, Y=0.1, p=0.5, order=1)

        self.assertTrue(ch.isTP, "2-qubit GAD channel should be TP")

    def test_gad_is_tp_single(self):
        """

        $Process.GAD(d=2, n=1, Y=0.1, p=0.5)$ is trace-preserving
        """

        ch = Process.GAD(d=2, n=1, Y=0.1, p=0.5, order=1)

        self.assertTrue(ch.isTP, "1-qubit GAD channel should be TP")

    def test_iid_ad_is_multiplex(self):
        """
        $IID.AD(n=2, d=2, y=0.1)$ returns a Multiplex of $n=2$ channels
        """


        mul = IID.AD(n=2, d=2, y=0.1)

        self.assertIsInstance(mul, Multiplex, "IID.AD should return Multiplex")

        self.assertEqual(len(mul.channels), 2, "IID.AD(n=2) should have 2 channels")

    def test_iid_ad_trace_preserved(self):
        """
        $IID.AD$ applied to $|00\\rangle\\langle 00|$ preserves trace
        """
        mul = IID.AD(n=2, d=2, y=0.1)
        rho = pt.zeros((4, 4), dtype=C64)

        rho[0, 0] = 1.0
        out = mul.run(rho)

        tr = float(pt.trace(out).real.item())

        self.assertAlmostEqual(tr, 1.0, places=4, msg="IID.AD should preserve trace")

    def test_iid_gad_is_multiplex(self):
        """
        $IID.GAD(n=2, d=2, y=0.1, p=0.5)$ returns a Multiplex of 2 channels
        """


        mul = IID.GAD(n=2, d=2, y=0.1, p=0.5)

        self.assertIsInstance(mul, Multiplex, "IID.GAD should return Multiplex")

        self.assertEqual(len(mul.channels), 2, "IID.GAD(n=2) should have 2 channels")

    def test_iid_gad_trace_preserved(self):
        """
        $IID.GAD$ applied to $|00\\rangle\\langle 00|$ preserves trace
        """
        mul = IID.GAD(n=2, d=2, y=0.1, p=0.5)
        rho = pt.zeros((4, 4), dtype=C64)

        rho[0, 0] = 1.0
        out = mul.run(rho)

        tr = float(pt.trace(out).real.item())

        self.assertAlmostEqual(tr, 1.0, places=4, msg="IID.GAD should preserve trace")

    def test_iid_ad_qutrit_trace_preserved(self):
        """
        $IID.AD$ for qutrits ($d=3$, $n=1$) preserves trace
        """
        mul = IID.AD(n=1, d=3, y=0.1)
        rho = pt.zeros((3, 3), dtype=C64)

        rho[0, 0] = 1.0
        out = mul.run(rho)

        tr = float(pt.trace(out).real.item())

        self.assertAlmostEqual(
            tr, 1.0, places=4, msg="IID.AD qutrit should preserve trace"
        )

    def test_ad_strong_damping(self):
        """
        $AD(Y=1.0)$ fully damps all excitations: $|1\\rangle\\langle1| \\to |0\\rangle\\langle0|$
        """
        ch = Process.AD(d=2, n=1, Y=1.0, order=1)
        rho1 = pt.zeros((2, 2), dtype=C64)
        rho1[1, 1] = 1.0

        out = ch.run(rho1)
        expected = rho0_qubit()

        diff = float(pt.norm(out - expected).real.item())

        self.assertAlmostEqual(
            diff, 0.0, places=4, msg="AD(Y=1) should fully damp |1> to |0>"
        )


if __name__ == "__main__":
    runner = Exam(
        name="Qudit Noise Tests",
        desc="Validation of channel properties (isTP, isCP, isCPTP, Choi, superop, Stinespring) and noise processes",
        file="noise.md",
    )
    runner.run(load(ChannelProperties))
    runner.run(load(ProcessChannels))
