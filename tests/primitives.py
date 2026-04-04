from MDR import Exam, load, Question
import sys

sys.path.append("..")

from qudit.random import random_unitary, random_state
from qudit import Basis, State
import numpy as np

Unity = lambda d: np.exp(2j * np.pi / d)


class BasisStates(Question):
    """
    Basis construction, State normalization, and density matrix tests.
    """

    def test_basis_notation(self):
        """
        $\\text{Basis}(2)(\\texttt{"00"}) = |00\\rangle = |0\\rangle \\otimes |0\\rangle$
        via string, integer, and tensor-product notations, two ways
        """
        Ket = Basis(2)
        K00 = Ket("00")
        self.stateEqual(Ket(0, 0), K00)
        self.stateEqual(Ket(0) ^ Ket(0), K00)

    def test_basis_amplitude(self):
        """
        $|00\\rangle[0] = 1$, all other amplitudes vanish
        """
        Ket = Basis(2)
        K00 = Ket("00")
        self.assertAlmostEqual(float(np.abs(K00[0])), 1.0, places=6)
        self.assertAlmostEqual(float(np.sum(np.abs(K00[1:]))), 0.0, places=6)

    def test_basis_orthogonal(self):
        """
        $\\langle i | j \\rangle = \\delta_{ij}$: computational basis states are orthonormal
        """
        Ket = Basis(3)
        for i in range(3):
            for j in range(3):
                inner = float(np.abs(np.vdot(Ket(i), Ket(j))))
                expected = 1.0 if i == j else 0.0
                self.assertAlmostEqual(inner, expected, places=6)

    def test_state_tr_normalized(self):
        """
        $\\langle \\psi | \\psi \\rangle = 1$ for any normalized State
        """
        Ket = Basis(2)
        psi = State(Ket(0) + Ket(1))  # State normalizes automatically
        self.assertAlmostEqual(psi.tr().real, 1.0, places=6)

    def test_state_density_two_ways(self):
        """
        $|\\psi\\rangle\\langle\\psi|$ via $\\texttt{.density()}$ and $\\texttt{np.outer}$
        agree, two ways
        """
        Ket = Basis(2)
        psi = State(Ket(0) + Ket(1))  # |+⟩ normalized
        rho_method = psi.density()
        rho_manual = np.outer(psi, psi.conj())
        np.testing.assert_allclose(rho_method, rho_manual, atol=1e-10)

    def test_state_pure(self):
        """
        $\\mathrm{Tr}(\\rho^2) = 1$ for a pure state; $\\texttt{isPure()}$ returns True
        """
        Ket = Basis(2)
        psi = State(Ket(0) + Ket(1))
        rho = psi.density()
        purity = float(np.real(np.trace(rho @ rho)))
        self.assertAlmostEqual(purity, 1.0, places=6)
        self.assertTrue(rho.isPure())

    def test_state_normalized(self):
        """
        Arbitrary superposition is a normalized complex vector of shape $(d^n,)$
        """
        pi = np.pi
        e, rt = np.exp, np.sqrt
        w, Ket = Unity(3), Basis(4)

        SV = State(
            w * Ket("0000")
            + w**2 * Ket("1010")
            + rt(3) * 1j * Ket("2010")
            + Ket("2200")
            + (9j + 16) * Ket("1210")
            + (w - w**2) * Ket("0022")
            + (w - 1) ** 2 * Ket("2020")
            + (e(1j * pi / 18) + 6) * Ket("2221")
            + Ket("0112")
            + (5 + 9j) * Ket("1200")
            + 0.67 * Ket("1111")
            + (9 * e(1j * pi / 16)) * Ket("2222")
        )

        self.assertEqual(SV.shape, (4**4,))
        self.assertTrue(np.iscomplexobj(SV))
        # Normalized: ||SV||² = 1
        self.assertAlmostEqual(float(np.vdot(SV, SV).real), 1.0, places=6)


class RandomStates(Question):
    """
    Random unitary and statevector generation tests (Haar-random distributions).
    """

    def test_unitary_isometry(self):
        """
        $U^\\dagger U = I$ for a Haar-random unitary (columns are orthonormal)
        """
        U = random_unitary(10)
        UdU = U.conj().T @ U
        np.testing.assert_allclose(UdU, np.eye(10), atol=1e-10)

    def test_state_normalized_norm(self):
        """
        $\\|\\psi\\| = 1$ for a Haar-random state
        """
        psi = random_state(20)
        self.assertAlmostEqual(float(np.linalg.norm(psi)), 1.0, places=10)

    def test_unitary_mean(self):
        """
        Haar-random $20 \\times 20$ unitary has near-zero mean across real and imaginary parts
        """
        U = random_unitary(20)
        all_vals = np.concatenate((U.real.flatten(), U.imag.flatten()))
        self.assertLess(abs(np.mean(all_vals)), 0.05)

    def test_state_mean(self):
        """
        Haar-random state of dimension 20 has near-zero mean across real and imaginary parts
        """
        S = random_state(20)
        all_vals = np.concatenate((S.real.flatten(), S.imag.flatten()))
        self.assertLess(abs(np.mean(all_vals)), 0.05)


if __name__ == "__main__":
    runner = Exam(
        name="Qudit Primitive Tests",
        desc="Validation of Basis, State, and Haar-random generation",
        file="primitives.md",
    )
    runner.run(load(BasisStates))
    runner.run(load(RandomStates))
