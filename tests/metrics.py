from MDR import Exam, load, Question
import sys

sys.path.append("..")

import numpy as np
from qudit.tools.metrics import Fidelity, Entropy, Info, Distance


class QuantumFidelity(Question):
    """
    Fidelity, channel, negativity, and information tests.
    """

    def test_fidelity_two_ways(self):
        """
        $F(|0\\rangle, |{+}\\rangle) = |\\langle 0|{+}\\rangle|^2 = \\tfrac{1}{2}$
        via function and direct overlap
        """
        psi = np.array([1, 0], dtype=complex)
        phi = np.array([1, 1], dtype=complex) / np.sqrt(2)
        f_func = Fidelity.default(psi, phi)
        f_manual = float(np.abs(np.vdot(psi, phi)) ** 2)

        self.assertAlmostEqual(f_func, f_manual, places=6, msg="Fidelity function and direct overlap should agree")

        self.assertAlmostEqual(f_func, 0.5, places=6, msg="F(|0>, |+>) should be 0.5")

    def test_fidelity_identical(self):
        """
        $F(|\\psi\\rangle, |\\psi\\rangle) = 1$
        """
        psi = np.array([1, 1], dtype=complex) / np.sqrt(2)

        self.assertAlmostEqual(Fidelity.default(psi, psi), 1.0, places=6, msg="F(|ψ>, |ψ>) should be 1")

    def test_fidelity_orthogonal(self):
        """
        $F(|0\\rangle, |1\\rangle) = 0$
        """
        self.assertAlmostEqual(
            Fidelity.default(
                np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex)
            ),
            0.0,
            places=6,
            msg="F(|0>, |1>) should be 0",
        )

    def test_fidelity_symmetric(self):
        """
        $F(\\rho, \\sigma) = F(\\sigma, \\rho)$ (symmetry of Uhlmann fidelity)
        """
        rho = np.diag([0.7, 0.3]).astype(complex)
        sigma = np.diag([0.4, 0.6]).astype(complex)

        self.assertAlmostEqual(
            Fidelity.default(rho, sigma), Fidelity.default(sigma, rho), places=6,
            msg="Fidelity should be symmetric",
        )

    def test_channel_two_ways(self):
        """
        $\Phi(\\rho) = \\sum_k K_k \\rho K_k^\\dagger = \\tfrac{I}{2}$
        computed via function and manual Kraus sum
        """
        K0 = np.sqrt(0.5) * np.eye(2)
        K1 = np.sqrt(0.5) * np.array([[0, 1], [1, 0]])
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        out_func = Fidelity.channel([K0, K1], rho)
        out_manual = K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T

        self.matEqual(out_func, out_manual, msg="Channel output should match manual Kraus sum")

        self.matEqual(out_func, 0.5 * np.eye(2), msg="Depolarizing channel output should be I/2")

    def test_channel_trace_preserving(self):
        """
        $\\mathrm{Tr}(\\Phi(\\rho)) = 1$ for any trace-preserving channel
        """
        K0 = np.sqrt(0.5) * np.eye(2)
        K1 = np.sqrt(0.5) * np.array([[0, 1], [1, 0]])
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        out = Fidelity.channel([K0, K1], rho)

        self.assertAlmostEqual(float(np.trace(out).real), 1.0, places=6, msg="Trace-preserving channel should have Tr(output) = 1")

    def test_negativity_bell(self):
        """
        $\\mathcal{N}(|\\Phi^+\\rangle) = 0.5$: eigenvalues of $\\rho^{T_B}$ are
        $\\{\\tfrac{1}{2}, \\tfrac{1}{2}, \\tfrac{1}{2}, -\\tfrac{1}{2}\\}$
        """
        bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        rho = np.outer(bell, bell.conj())

        self.assertAlmostEqual(Fidelity.negativity(rho, 2, 2), 0.5, places=6, msg="Bell state negativity should be 0.5")

    def test_negativity_separable(self):
        """
        $\\mathcal{N}(\\rho_A \\otimes \\rho_B) = 0$ for any separable product state
        """
        rho_A = np.diag([0.7, 0.3]).astype(complex)
        rho_B = np.diag([0.4, 0.6]).astype(complex)
        rho_sep = np.kron(rho_A, rho_B)

        self.assertAlmostEqual(Fidelity.negativity(rho_sep, 2, 2), 0.0, places=6, msg="Separable state negativity should be 0")

    def test_mutual_info_two_ways(self):
        """
        $I(A:B) = S(A) + S(B) - S(AB) = 2$ bits for $|\\Phi^+\\rangle$,
        computed via function and manual partial traces
        """
        bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        rho_AB = np.outer(bell, bell.conj())
        I_func = Info.mutual(rho_AB, 2, 2)
        # Manual: partial trace without library
        r = rho_AB.reshape(2, 2, 2, 2)
        rho_A = np.trace(r, axis1=1, axis2=3)
        rho_B = np.trace(r, axis1=0, axis2=2)
        I_manual = (
            Entropy.neumann(rho_A) + Entropy.neumann(rho_B) - Entropy.neumann(rho_AB)
        )

        self.assertAlmostEqual(I_func, I_manual, places=5, msg="Mutual info function and manual calculation should agree")

        self.assertAlmostEqual(I_func, 2.0, places=5, msg="Bell state mutual info should be 2 bits")

    def test_mutual_info_product(self):
        """
        $I(A:B) = 0$ for product state $\\rho_A \\otimes \\rho_B$
        """
        rho_A = np.diag([0.7, 0.3]).astype(complex)
        rho_B = np.diag([0.5, 0.5]).astype(complex)
        rho_prod = np.kron(rho_A, rho_B)

        self.assertAlmostEqual(Info.mutual(rho_prod, 2, 2), 0.0, places=5, msg="Product state mutual info should be 0")

    def test_coherent_info_bell(self):
        """
        $I_c(A\\rangle B) = S(B) - S(AB) = 1$ bit for $|\\Phi^+\\rangle$
        """
        bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        rho = np.outer(bell, bell.conj())

        self.assertAlmostEqual(Info.coherent(rho, 2, 2), 1.0, places=5, msg="Bell state coherent info should be 1 bit")


class QuantumEntropy(Question):
    """
    Entropy functional tests: von Neumann, Shannon, Tsallis, Rényi, Hartley,
    unified, relative, and conditional entropy.
    """

    def test_neumann_pure_zero(self):
        """
        $S(|\\psi\\rangle\\langle\\psi|) = 0$: pure states have zero entropy
        """
        rho = np.array([[1, 0], [0, 0]], dtype=complex)

        self.assertAlmostEqual(Entropy.neumann(rho), 0.0, places=6, msg="Pure state von Neumann entropy should be 0")

    def test_neumann_maximally_mixed(self):
        """
        $S(I/d) = \\log_2 d$: maximally mixed qubit has entropy 1 bit
        """
        rho = 0.5 * np.eye(2, dtype=complex)

        self.assertAlmostEqual(Entropy.neumann(rho), 1.0, places=6, msg="Maximally mixed qubit entropy should be 1 bit")

    def test_neumann_vs_shannon(self):
        """
        $S(\\mathrm{diag}(p_i)) = H(p_i)$: von Neumann on diagonal $\\rho$ equals Shannon entropy,
        two ways
        """
        p = np.array([0.7, 0.3])
        rho = np.diag(p).astype(complex)

        self.assertAlmostEqual(Entropy.neumann(rho), Entropy.shannon(p), places=6, msg="von Neumann on diagonal rho should equal Shannon entropy")

    def test_neumann_value(self):
        """
        $S(\\mathrm{diag}(0.7, 0.3)) = -0.7\\log_2 0.7 - 0.3\\log_2 0.3 \\approx 0.8813$
        """
        rho = np.diag([0.7, 0.3]).astype(complex)
        expected = -0.7 * np.log2(0.7) - 0.3 * np.log2(0.3)

        self.assertAlmostEqual(Entropy.neumann(rho), expected, places=6, msg="von Neumann entropy value mismatch for diag(0.7, 0.3)")

    def test_shannon_uniform(self):
        """
        $H(1/n, \\ldots, 1/n) = \\log_2 n$ bits
        """
        p = np.full(4, 0.25)

        self.assertAlmostEqual(Entropy.shannon(p), np.log2(4), places=6, msg="Uniform Shannon entropy should be log2(4)=2 bits")

    def test_tsallis_value(self):
        """
        $S_2(\\mathrm{diag}(0.6, 0.4)) = \\frac{1 - (0.6^2 + 0.4^2)}{1} = 0.48$
        """
        rho = np.diag([0.6, 0.4]).astype(complex)

        self.assertAlmostEqual(Entropy.tsallis(rho, q=2), 0.48, places=6, msg="Tsallis(q=2) entropy value mismatch")

    def test_renyi_value(self):
        """
        $S_2(\\mathrm{diag}(0.6, 0.4)) = -\\log_2(0.6^2 + 0.4^2) \\approx 0.9437$
        """
        rho = np.diag([0.6, 0.4]).astype(complex)
        expected = -np.log2(0.6**2 + 0.4**2)

        self.assertAlmostEqual(Entropy.renyi(rho, alpha=2), expected, places=6, msg="Renyi(alpha=2) entropy value mismatch")

    def test_hartley_value(self):
        """
        $H_0(0.25, 0.25, 0.25, 0.25) = \\log_2 4 = 2$ bits
        """
        self.assertAlmostEqual(
            Entropy.hartley(np.array([0.25, 0.25, 0.25, 0.25])), 2.0, places=6,
            msg="Hartley entropy for 4 uniform outcomes should be 2 bits",
        )

    def test_relative_zero(self):
        """
        $D(\\rho \\| \\rho) = 0$
        """
        rho = np.diag([0.7, 0.3]).astype(complex)

        self.assertAlmostEqual(Entropy.relative(rho, rho), 0.0, places=5, msg="Relative entropy D(rho||rho) should be 0")

    def test_relative_value(self):
        """
        $D(\\mathrm{diag}(0.8,0.2) \\| \\mathrm{diag}(0.5,0.5)) =
        \\sum_i p_i \\log_2(p_i/q_i) \\approx 0.2781$ bits
        """
        rho = np.diag([0.8, 0.2]).astype(complex)
        sigma = np.diag([0.5, 0.5]).astype(complex)
        expected = 0.8 * np.log2(0.8 / 0.5) + 0.2 * np.log2(0.2 / 0.5)

        self.assertAlmostEqual(Entropy.relative(rho, sigma), expected, places=5, msg="Relative entropy value mismatch")

    def test_unified_reduces_to_renyi(self):
        """
        $S^{(q,\\alpha)}$ at $q=1$ reduces to $S_\\alpha$ (Rényi), two ways
        """
        rho = np.diag([0.6, 0.4]).astype(complex)

        self.assertAlmostEqual(
            Entropy.unified(rho, q=1.0, alpha=2.0),
            Entropy.renyi(rho, alpha=2.0),
            places=6,
            msg="Unified entropy at q=1 should equal Renyi entropy",
        )

    def test_unified_reduces_to_tsallis(self):
        """
        $S^{(q,\\alpha)}$ at $\\alpha=1$ reduces to $S_q$ (Tsallis), two ways
        """
        rho = np.diag([0.6, 0.4]).astype(complex)

        self.assertAlmostEqual(
            Entropy.unified(rho, q=2.0, alpha=1.0),
            Entropy.tsallis(rho, q=2.0),
            places=6,
            msg="Unified entropy at alpha=1 should equal Tsallis entropy",
        )

    def test_conditional_product(self):
        """
        $S(A|B) = S(A)$ for product state $\\rho_A \\otimes \\rho_B$
        (conditioning on independent $B$ leaves entropy of $A$ unchanged)
        """
        rho_A = np.diag([0.7, 0.3]).astype(complex)
        rho_B = np.diag([0.5, 0.5]).astype(complex)
        rho_prod = np.kron(rho_A, rho_B)
        S_A = Entropy.neumann(rho_A)

        self.assertAlmostEqual(Entropy.conditional(rho_prod, 2, 2), S_A, places=5, msg="S(A|B) for product state should equal S(A)")


class QuantumDistance(Question):
    """
    Distance measure tests: trace distance, Bures distance, Jensen-Shannon divergence.
    """

    def test_trace_identical(self):
        """
        $T(\\rho, \\rho) = 0$
        """
        rho = np.diag([0.7, 0.3]).astype(complex)

        self.assertAlmostEqual(Distance.trace(rho, rho), 0.0, places=6, msg="Trace distance T(rho, rho) should be 0")

    def test_trace_value(self):
        """
        $T(|0\\rangle\\langle 0|, |{+}\\rangle\\langle{+}|)
        = \\sqrt{1 - F} = \\sqrt{1/2} \\approx 0.7071$
        """
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        sigma = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        expected = np.sqrt(0.5)

        self.assertAlmostEqual(Distance.trace(rho, sigma), expected, places=5, msg="Trace distance value mismatch")

    def test_trace_two_ways(self):
        """
        $T(\\rho, \\sigma) = \\tfrac{1}{2}\\sum_i|\\lambda_i(\\rho-\\sigma)|$
        via function and eigenvalue sum
        """
        rho = np.diag([0.8, 0.2]).astype(complex)
        sigma = np.diag([0.5, 0.5]).astype(complex)
        t_func = Distance.trace(rho, sigma)
        # Manual: half sum of absolute eigenvalues of (rho - sigma)
        evals = np.linalg.eigvalsh(rho - sigma)
        t_manual = 0.5 * np.sum(np.abs(evals))

        self.assertAlmostEqual(t_func, t_manual, places=5, msg="Trace distance function and manual eigenvalue sum should agree")

    def test_bures_identical(self):
        """
        $D_B(\\rho, \\rho) = 0$
        """
        rho = np.diag([0.7, 0.3]).astype(complex)

        self.assertAlmostEqual(Distance.bures(rho, rho), 0.0, places=6, msg="Bures distance D_B(rho, rho) should be 0")

    def test_bures_value(self):
        """
        $D_B(|0\\rangle, |{+}\\rangle) = \\sqrt{2 - \\sqrt{2}} \\approx 0.7654$
        """
        psi = np.array([1, 0], dtype=complex)
        phi = np.array([1, 1], dtype=complex) / np.sqrt(2)
        expected = np.sqrt(2 - np.sqrt(2))

        self.assertAlmostEqual(Distance.bures(psi, phi), expected, places=5, msg="Bures distance value mismatch for |0> vs |+>")

    def test_bures_vs_fidelity(self):
        """
        $D_B(\\rho, \\sigma) = \\sqrt{2 - 2\\sqrt{F(\\rho,\\sigma)}}$:
        Bures distance and fidelity are consistent, two ways
        """
        rho = np.diag([0.8, 0.2]).astype(complex)
        sigma = np.diag([0.5, 0.5]).astype(complex)
        F = Fidelity.default(rho, sigma)
        bures_manual = np.sqrt(2 - 2 * np.sqrt(F))

        self.assertAlmostEqual(Distance.bures(rho, sigma), bures_manual, places=5, msg="Bures distance and fidelity should be consistent")


if __name__ == "__main__":
    runner = Exam(
        name="Quantum Metrics Tests",
        desc="Validation of fidelity, entropy, information, and distance measures",
        file="metrics.md",
    )
    runner.run(load(QuantumFidelity))
    runner.run(load(QuantumEntropy))
    runner.run(load(QuantumDistance))
