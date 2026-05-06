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


class ProcessLibraryTests(Question):
    """
    Process.* constructors: CPTP checks across channel types and dimensions.
    """

    def test_depolarising_qubit_is_cptp(self):
        """
        $Process.Depolarising(d=2, n=1, p=0.1)$ is CPTP
        """
        ch = Process.Depolarising(d=2, n=1, p=0.1)

        self.assertTrue(ch.isCPTP, "Depolarising qubit channel should be CPTP")

    def test_phasedamp_qubit_is_cptp(self):
        """
        $Process.PhaseDamp(d=2, n=1, p=0.1)$ is CPTP
        """
        ch = Process.PhaseDamp(d=2, n=1, p=0.1)

        self.assertTrue(ch.isCPTP, "PhaseDamp qubit channel should be CPTP")

    def test_reset_qubit_is_cptp(self):
        """
        $Process.Reset(d=2, n=1, p=0.1)$ is CPTP
        """
        ch = Process.Reset(d=2, n=1, p=0.1)

        self.assertTrue(ch.isCPTP, "Reset qubit channel should be CPTP")

    def test_depolarising_as_weyl_channel_is_cptp(self):
        """
        $Process.Depolarising(d=2, n=1, p=0.1)$ (Weyl-based) is CPTP
        """
        ch = Process.Depolarising(d=2, n=1, p=0.1)

        self.assertTrue(ch.isCPTP, "Weyl/Depolarising qubit channel should be CPTP")

    def test_ad_qutrit_is_cptp(self):
        """
        $Process.AD(d=3, n=1, Y=0.1, order=2)$ qutrit amplitude damping is CPTP
        """
        ch = Process.AD(d=3, n=1, Y=0.1, order=2)

        self.assertTrue(ch.isCPTP, "Qutrit AD channel should be CPTP")

    def test_depolarising_qutrit_is_cptp(self):
        """
        $Process.Depolarising(d=3, n=1, p=0.05)$ qutrit depolarising is CPTP
        """
        ch = Process.Depolarising(d=3, n=1, p=0.05)

        self.assertTrue(ch.isCPTP, "Qutrit depolarising channel should be CPTP")

    def test_pauli_is_cptp(self):
        """
        $Process.Pauli(n=1, p=[0.1, 0.05, 0.05])$ is CPTP
        """
        ch = Process.Pauli(n=1, p=[0.1, 0.05, 0.05])

        self.assertTrue(ch.isCPTP, "Pauli channel should be CPTP")

    def test_thermalrelax_is_cptp(self):
        """
        $Process.ThermalRelax(n=1, T1=100, T2=80, t=1)$ is CPTP
        """
        ch = Process.ThermalRelax(n=1, T1=100.0, T2=80.0, t=1.0)

        self.assertTrue(ch.isCPTP, "ThermalRelax channel should be CPTP")


class ChannelOutputTests(Question):
    """
    Verify channel.run() returns a valid density matrix: Hermitian, PSD, trace-1.
    """

    def _rho0(self):
        r = pt.zeros((2, 2), dtype=C64)
        r[0, 0] = 1.0

        return r

    def test_ad_output_hermitian(self):
        """
        $\\Phi_{AD}(|0\\rangle\\langle0|)$ is Hermitian: $\\rho^\\dagger = \\rho$
        """
        ch = Process.AD(d=2, n=1, Y=0.1, order=2)
        out = ch.run(self._rho0())

        diff = float(pt.norm(out - out.conj().T).real.item())

        self.assertAlmostEqual(diff, 0.0, places=4, msg="AD output should be Hermitian")

    def test_ad_output_psd(self):
        """
        $\\Phi_{AD}(|0\\rangle\\langle0|)$ is PSD: all eigenvalues $\\geq 0$
        """
        ch = Process.AD(d=2, n=1, Y=0.1, order=2)
        out = ch.run(self._rho0())

        evals = pt.linalg.eigvalsh(out.to(pt.float64))

        self.assertTrue(
            bool((evals >= -1e-6).all().item()),
            "AD output should be positive semi-definite",
        )

    def test_depolarising_output_trace_one(self):
        """
        $\\mathrm{Tr}(\\Phi_{Dep}(|0\\rangle\\langle0|)) = 1$
        """
        ch = Process.Depolarising(d=2, n=1, p=0.1)
        out = ch.run(self._rho0())

        tr = float(pt.trace(out).real.item())

        self.assertAlmostEqual(
            tr, 1.0, places=4, msg="Depolarising output trace should be 1"
        )

    def test_pauli_mixed_state_invariant(self):
        """
        $\\Phi_{Pauli}(I/2) = I/2$: balanced Pauli channel leaves maximally mixed state invariant
        """
        p = [1.0 / 6, 1.0 / 6, 1.0 / 6]
        ch = Process.Pauli(n=1, p=p)

        rho_mix = rho_mixed_qubit()
        out = ch.run(rho_mix)

        diff = float(pt.norm(out - rho_mix).real.item())

        self.assertAlmostEqual(
            diff,
            0.0,
            places=3,
            msg="Pauli channel should leave maximally mixed state invariant",
        )

    def test_reset_full_collapses_to_ground(self):
        """
        $Process.Reset(d=2, n=1, p=1.0)$ collapses any state to $|0\\rangle\\langle0|$
        """
        ch = Process.Reset(d=2, n=1, p=1.0)

        rho1 = pt.zeros((2, 2), dtype=C64)
        rho1[1, 1] = 1.0
        out = ch.run(rho1)

        diff = float(pt.norm(out - self._rho0()).real.item())

        self.assertAlmostEqual(
            diff, 0.0, places=4, msg="Full reset should collapse to |0><0|"
        )

    def test_phasedamp_preserves_populations(self):
        """
        PhaseDamp preserves diagonal populations: $\\rho_{00}$ and $\\rho_{11}$ unchanged
        """
        ch = Process.PhaseDamp(d=2, n=1, p=0.5)

        rho = pt.tensor([[0.6, 0.4], [0.4, 0.4]], dtype=C64)
        out = ch.run(rho)

        self.assertAlmostEqual(
            float(out[0, 0].real.item()),
            0.6,
            places=4,
            msg="PhaseDamp should not change |0><0| population",
        )

        self.assertAlmostEqual(
            float(out[1, 1].real.item()),
            0.4,
            places=4,
            msg="PhaseDamp should not change |1><1| population",
        )

    def test_phasedamp_kills_coherence(self):
        """
        PhaseDamp($p=1$) fully kills off-diagonal coherences
        """
        ch = Process.PhaseDamp(d=2, n=1, p=1.0)

        rho = pt.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=C64)
        out = ch.run(rho)

        self.assertAlmostEqual(
            float(pt.abs(out[0, 1]).item()),
            0.0,
            places=4,
            msg="Full phase damping should eliminate off-diagonal elements",
        )


class MultiplexTests(Question):
    """
    IID channel constructors and Multiplex.run() trace-preservation.
    """

    def test_iid_ad_n2_is_multiplex(self):
        """
        $IID.AD(n=2, d=2, y=0.1)$ returns a Multiplex instance
        """
        mul = IID.AD(n=2, d=2, y=0.1)

        self.assertIsInstance(mul, Multiplex, "IID.AD should return a Multiplex")

    def test_iid_ad_n2_run_trace_preserved(self):
        """
        $IID.AD(n=2)$.run$(\\rho)$ on a 2-qubit state has $\\mathrm{Tr}=1$
        """
        mul = IID.AD(n=2, d=2, y=0.1)

        rho = pt.zeros((4, 4), dtype=C64)
        rho[0, 0] = 1.0
        out = mul.run(rho)

        tr = float(pt.trace(out).real.item())

        self.assertAlmostEqual(
            tr, 1.0, places=4, msg="IID.AD n=2 should preserve trace"
        )

    def test_iid_ad_n3_trace_preserved(self):
        """
        $IID.AD(n=3, d=2, y=0.1)$.run$(\\rho)$ on a 3-qubit state has $\\mathrm{Tr}=1$
        """
        mul = IID.AD(n=3, d=2, y=0.1)

        rho = pt.zeros((8, 8), dtype=C64)
        rho[0, 0] = 1.0
        out = mul.run(rho)

        tr = float(pt.trace(out).real.item())

        self.assertAlmostEqual(
            tr, 1.0, places=4, msg="IID.AD n=3 should preserve trace"
        )

    def test_iid_depolarising_n2_trace_preserved(self):
        """
        $IID.Depolarising(n=2, d=2, p=0.1)$.run$(\\rho)$ has $\\mathrm{Tr}=1$
        """
        mul = IID.Depolarising(n=2, d=2, p=0.1)

        rho = pt.zeros((4, 4), dtype=C64)
        rho[0, 0] = 1.0
        out = mul.run(rho)

        tr = float(pt.trace(out).real.item())

        self.assertAlmostEqual(
            tr, 1.0, places=4, msg="IID.Depolarising n=2 should preserve trace"
        )

    def test_iid_reset_n2_trace_preserved(self):
        """
        $IID.Reset(n=2, d=2, p=0.3)$.run$(\\rho)$ has $\\mathrm{Tr}=1$
        """
        mul = IID.Reset(n=2, d=2, p=0.3)

        rho = pt.zeros((4, 4), dtype=C64)
        rho[3, 3] = 1.0
        out = mul.run(rho)

        tr = float(pt.trace(out).real.item())

        self.assertAlmostEqual(
            tr, 1.0, places=4, msg="IID.Reset n=2 should preserve trace"
        )

    def test_multiplex_length_matches_n(self):
        """
        $IID.AD(n=3)$ Multiplex has exactly 3 channels
        """
        mul = IID.AD(n=3, d=2, y=0.1)

        self.assertEqual(len(mul.channels), 3, "IID.AD(n=3) should have 3 channels")


class ChannelCompositionTests(Question):
    """
    Sequential channel application and identity-channel behaviour.
    """

    def _fidelity(self, rho: pt.Tensor) -> float:
        return float(rho[0, 0].real.item())

    def test_double_ad_more_noisy(self):
        """
        Applying AD twice gives lower fidelity to $|0\\rangle\\langle0|$ than once
        """
        rho1 = pt.zeros((2, 2), dtype=C64)
        rho1[1, 1] = 1.0

        ch = Process.AD(d=2, n=1, Y=0.5, order=1)

        out1 = ch.run(rho1)
        out2 = ch.run(out1)

        f1 = self._fidelity(out1)
        f2 = self._fidelity(out2)

        self.assertGreater(
            f2, f1, msg="Applying AD twice should increase ground-state population"
        )

    def test_low_noise_pauli_near_identity(self):
        """
        $Process.Pauli(n=1, p=[10^{-4}, 10^{-4}, 10^{-4}])$ output is close to input
        """
        eps = 1e-4
        ch = Process.Pauli(n=1, p=[eps, eps, eps])

        rho = pt.tensor([[0.7, 0.3], [0.3, 0.3]], dtype=C64)
        out = ch.run(rho)

        diff = float(pt.norm(out - rho).real.item())

        self.assertLess(
            diff, 1e-2, msg="Near-zero Pauli noise should leave state nearly unchanged"
        )

    def test_low_noise_depolarising_near_identity(self):
        """
        $Process.Depolarising(d=2, n=1, p=10^{-4})$ output is close to input
        """
        ch = Process.Depolarising(d=2, n=1, p=1e-4)

        rho = rho_mixed_qubit()
        out = ch.run(rho)

        diff = float(pt.norm(out - rho).real.item())

        self.assertLess(
            diff, 1e-2, msg="Near-zero depolarising should leave state nearly unchanged"
        )

    def test_ad_then_reset_trace_preserved(self):
        """
        Sequential AD then Reset preserves trace: $\\mathrm{Tr}((\\Phi_{RS} \\circ \\Phi_{AD})(\\rho)) = 1$
        """
        ch_ad = Process.AD(d=2, n=1, Y=0.3, order=1)
        ch_rs = Process.Reset(d=2, n=1, p=0.5)

        rho = pt.zeros((2, 2), dtype=C64)
        rho[1, 1] = 1.0

        out = ch_rs.run(ch_ad.run(rho))

        tr = float(pt.trace(out).real.item())

        self.assertAlmostEqual(
            tr, 1.0, places=4, msg="Sequential AD+Reset should preserve trace"
        )

    def test_higher_noise_ad_more_damping(self):
        """
        Higher $Y$ in AD channel yields greater damping of $|1\\rangle\\langle1|$ population
        """
        rho1 = pt.zeros((2, 2), dtype=C64)
        rho1[1, 1] = 1.0

        ch_low = Process.AD(d=2, n=1, Y=0.1, order=1)
        ch_high = Process.AD(d=2, n=1, Y=0.9, order=1)

        pop_low = float(ch_low.run(rho1)[1, 1].real.item())
        pop_high = float(ch_high.run(rho1)[1, 1].real.item())

        self.assertGreater(
            pop_low, pop_high, msg="Higher Y should damp excited population more"
        )


if __name__ == "__main__":
    runner = Exam(
        name="Qudit Noise Tests",
        desc="Validation of channel properties (isTP, isCP, isCPTP, Choi, superop, Stinespring) and noise processes",
        file="noise.md",
    )
    runner.run(load(ChannelProperties))
    runner.run(load(ProcessChannels))
    runner.run(load(ProcessLibraryTests))
    runner.run(load(ChannelOutputTests))
    runner.run(load(MultiplexTests))
    runner.run(load(ChannelCompositionTests))
