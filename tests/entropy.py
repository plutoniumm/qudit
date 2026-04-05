from MDR import Exam, load, Question
import sys
import numpy as np

sys.path.append("..")

from qudit.tools.metrics import Fidelity, Entropy, Distance, Info
from qudit.utils import partial


def pure(v):
    v = np.array(v, dtype=complex)
    v /= np.linalg.norm(v)

    return np.outer(v, v.conj())


ket0 = np.array([1.0, 0.0], dtype=complex)
ket1 = np.array([0.0, 1.0], dtype=complex)
bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
rho0 = pure(ket0)
rho1 = pure(ket1)
rho_bell = pure(bell)
rho_mixed = np.eye(2, dtype=complex) / 2


class EntropyTests(Question):
    """
    Von Neumann, Shannon, Tsallis, Renyi, Hartley, relative, conditional entropies.
    """

    def test_neumann_pure(self):
        """
        Pure state has zero von Neumann entropy
        """
        self.assertAlmostEqual(
            Entropy.neumann(rho0),
            0.0,
            places=6,
            msg="Pure state von Neumann entropy should be 0",
        )

    def test_neumann_mixed(self):
        """
        Maximally mixed qubit has entropy = 1 ebit
        """

        self.assertAlmostEqual(
            Entropy.neumann(rho_mixed),
            1.0,
            places=5,
            msg="Maximally mixed qubit entropy should be 1 ebit",
        )

    def test_neumann_bell_reduced(self):
        """
        Bell state reduced to one qubit is maximally mixed: entropy = 1
        """

        rho_A = partial.trace(rho_bell, 2, 2, keep="A")

        self.assertAlmostEqual(
            Entropy.neumann(np.array(rho_A)),
            1.0,
            places=5,
            msg="Bell state reduced entropy should be 1",
        )

    def test_shannon_uniform(self):
        """
        Shannon entropy of [0.5, 0.5] = 1 bit
        """
        self.assertAlmostEqual(
            Entropy.shannon(np.array([0.5, 0.5])),
            1.0,
            places=6,
            msg="Shannon entropy of [0.5, 0.5] should be 1 bit",
        )

    def test_shannon_deterministic(self):
        """
        Shannon entropy of [1.0] = 0
        """
        self.assertAlmostEqual(
            Entropy.shannon(np.array([1.0])),
            0.0,
            places=6,
            msg="Shannon entropy of [1.0] should be 0",
        )

    def test_tsallis_q1_equals_neumann(self):
        """
        Tsallis entropy at q=1 equals von Neumann entropy
        """
        ts = Entropy.tsallis(rho_mixed, q=1.0)

        vn = Entropy.neumann(rho_mixed)

        self.assertAlmostEqual(
            ts, vn, places=4, msg="Tsallis(q=1) should equal von Neumann entropy"
        )

    def test_renyi_alpha1_equals_neumann(self):
        """
        Renyi entropy at alpha=1 equals von Neumann entropy
        """
        re = Entropy.renyi(rho_mixed, alpha=1.0)

        vn = Entropy.neumann(rho_mixed)

        self.assertAlmostEqual(
            re, vn, places=4, msg="Renyi(alpha=1) should equal von Neumann entropy"
        )

    def test_renyi_pure(self):
        """
        Renyi entropy of pure state = 0 for any alpha
        """
        self.assertAlmostEqual(
            Entropy.renyi(rho0, alpha=2.0),
            0.0,
            places=6,
            msg="Renyi entropy of pure state should be 0",
        )

    def test_hartley(self):
        """
        Hartley entropy = log2(|support|): uniform over 4 outcomes = 2 bits
        """
        self.assertAlmostEqual(
            Entropy.hartley(np.array([0.25, 0.25, 0.25, 0.25])),
            2.0,
            places=6,
            msg="Hartley entropy for 4 uniform outcomes should be 2 bits",
        )

    def test_relative_self(self):
        """
        Relative entropy D(p‖p) = 0
        """
        self.assertAlmostEqual(
            Entropy.relative(rho_mixed, rho_mixed),
            0.0,
            places=4,
            msg="Relative entropy D(p||p) should be 0",
        )

    def test_conditional_pure_bipartite(self):
        """
        For pure bipartite state, S(A|B) = -S(B) (negative for entangled)
        """
        S_AB = Entropy.neumann(rho_bell)
        rho_B = partial.trace(rho_bell, 2, 2, keep="B")
        S_B = Entropy.neumann(np.array(rho_B))

        cond = Entropy.conditional(rho_bell, 2, 2)

        self.assertAlmostEqual(
            cond,
            S_AB - S_B,
            places=5,
            msg="S(A|B) = S(AB) - S(B) for pure bipartite state",
        )


class FidelityTests(Question):
    """
    Uhlmann fidelity and related measures.
    """

    def test_fidelity_self(self):
        """
        F(|ψ⟩,|ψ⟩) = 1
        """
        self.assertAlmostEqual(
            Fidelity.default(ket0, ket0), 1.0, places=6, msg="F(|ψ>,|ψ>) should be 1"
        )

    def test_fidelity_orthogonal(self):
        """
        F(|0⟩,|1⟩) = 0
        """
        self.assertAlmostEqual(
            Fidelity.default(ket0, ket1), 0.0, places=6, msg="F(|0>,|1>) should be 0"
        )

    def test_fidelity_mixed_self(self):
        """
        F(p,p) = 1 for any state
        """
        self.assertAlmostEqual(
            Fidelity.default(rho_mixed, rho_mixed),
            1.0,
            places=5,
            msg="F(rho, rho) should be 1 for any state",
        )


class DistanceTests(Question):
    """
    Trace distance, Bures distance, Jensen-Shannon divergence.
    """

    def test_trace_self(self):
        """
        T(p,p) = 0
        """
        self.assertAlmostEqual(
            Distance.trace(rho0, rho0),
            0.0,
            places=6,
            msg="Trace distance T(rho, rho) should be 0",
        )

    def test_trace_orthogonal(self):
        """
        T(|0⟩⟨0|,|1⟩⟨1|) = 1
        """
        self.assertAlmostEqual(
            Distance.trace(rho0, rho1),
            1.0,
            places=5,
            msg="Trace distance between orthogonal pure states should be 1",
        )

    def test_bures_self(self):
        """
        Bures distance D_B(p,p) = 0
        """
        self.assertAlmostEqual(
            Distance.bures(rho0, rho0),
            0.0,
            places=6,
            msg="Bures distance D_B(rho, rho) should be 0",
        )

    def test_bures_orthogonal(self):
        """
        Bures distance between orthogonal pure states = √2
        """
        self.assertAlmostEqual(
            Distance.bures(rho0, rho1),
            np.sqrt(2),
            places=5,
            msg="Bures distance between orthogonal pure states should be √2",
        )

    def test_jensen_shannon_self(self):
        """
        Jensen-Shannon divergence JSD(p‖p) = 0
        """
        self.assertAlmostEqual(
            Distance.jensen_shannon(rho0, rho0),
            0.0,
            places=4,
            msg="Jensen-Shannon JSD(p||p) should be 0",
        )


class InfoTests(Question):
    """
    Mutual information and coherent information.
    """

    def test_mutual_product(self):
        """
        I(A:B) = 0 for a product state
        """

        rho_prod = np.kron(rho0, rho1)

        self.assertAlmostEqual(
            Info.mutual(rho_prod, 2, 2),
            0.0,
            places=5,
            msg="Mutual info I(A:B) for product state should be 0",
        )

    def test_mutual_bell(self):
        """
        I(A:B) = 2 for a Bell state (maximally entangled)
        """
        self.assertAlmostEqual(
            Info.mutual(rho_bell, 2, 2),
            2.0,
            places=5,
            msg="Bell state mutual info should be 2 bits",
        )

    def test_coherent_pure(self):
        """
        Coherent info I_c(A⟩B) = S(B) - S(AB) = 1 - 0 = 1 for Bell state
        """
        self.assertAlmostEqual(
            Info.coherent(rho_bell, 2, 2),
            1.0,
            places=5,
            msg="Bell state coherent info should be 1",
        )


class InfoCondtionalTests(Question):
    """
    Info.conditional (measurement-based conditional entropy) and cross-checks.

    """

    def test_conditional_classical_product(self):
        """
        Measurement-based $S(B|A)$ on product state $\\rho_A \\otimes \\rho_B$
        equals $S(\\rho_B)$: measuring $A$ on a product state leaves $B$ unchanged
        """
        from qudit.tools.metrics import Info, Entropy

        rho_A = np.diag([0.5, 0.5]).astype(complex)
        rho_B = np.diag([0.7, 0.3]).astype(complex)
        rho_prod = np.kron(rho_A, rho_B)
        cond_true = Info.conditional(rho_prod, 2, 2, true_case=True)

        S_B = Entropy.neumann(rho_B)

        self.assertAlmostEqual(
            cond_true,
            S_B,
            places=4,
            msg="Measurement-based conditional S(B|A) for product = S(B)",
        )

    def test_conditional_standard_product_equals_sa(self):
        """
        $S(A|B) = S(A)$ for product state $\\rho_A \\otimes \\rho_B$:
        standard conditional entropy (not measurement-based) equals $S(A)$
        """
        from qudit.tools.metrics import Info, Entropy

        rho_A = np.diag([0.7, 0.3]).astype(complex)
        rho_B = np.diag([0.5, 0.5]).astype(complex)
        rho_prod = np.kron(rho_A, rho_B)
        cond_standard = Info.conditional(rho_prod, 2, 2, true_case=False)

        S_A = Entropy.neumann(rho_A)

        self.assertAlmostEqual(
            cond_standard,
            S_A,
            places=4,
            msg="S(A|B) for product state should equal S(A)",
        )

    def test_mutual_info_nonnegative(self):
        """
        $I(A:B) \\geq 0$ for any bipartite state (Klein's inequality)

        """
        from qudit.tools.metrics import Info

        # Mixed entangled state
        bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)

        rho = np.outer(bell, bell.conj())

        self.assertGreaterEqual(
            Info.mutual(rho, 2, 2), 0.0, msg="Mutual information must be non-negative"
        )

    def test_coherent_info_value(self):
        """
        $I_c(A\\rangle B) = S(B) - S(AB)$: coherent information for Bell state.
        $S(B) = 1$, $S(AB) = 0$, so $I_c = 1$.
        """
        from qudit.tools.metrics import Info

        bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        rho = np.outer(bell, bell.conj())

        ic = Info.coherent(rho, 2, 2)

        self.assertAlmostEqual(
            ic, 1.0, places=5, msg="Coherent information of Bell state should be 1"
        )


class DistanceExtendedTests(Question):
    """
    Extended distance tests: Jensen-Shannon divergence and relative entropy with known values.
    """

    def test_jensen_shannon_nonneg(self):
        """
        $JSD(\\rho \\| \\sigma) \\geq 0$: Jensen-Shannon divergence is non-negative
        """
        from qudit.tools.metrics import Distance

        rho0 = np.array([[1, 0], [0, 0]], dtype=complex)

        rho1 = np.array([[0, 0], [0, 1]], dtype=complex)

        self.assertGreaterEqual(
            Distance.jensen_shannon(rho0, rho1),
            0.0,
            msg="Jensen-Shannon divergence should be non-negative",
        )

    def test_jensen_shannon_symmetric(self):
        """
        $JSD(\\rho \\| \\sigma) = JSD(\\sigma \\| \\rho)$: Jensen-Shannon divergence is symmetric

        """
        from qudit.tools.metrics import Distance

        rho = np.diag([0.7, 0.3]).astype(complex)

        sigma = np.diag([0.4, 0.6]).astype(complex)

        self.assertAlmostEqual(
            Distance.jensen_shannon(rho, sigma),
            Distance.jensen_shannon(sigma, rho),
            places=5,
            msg="Jensen-Shannon divergence should be symmetric",
        )

    def test_relative_entropy_distance_same_as_entropy_relative(self):
        """
        $Distance.relative\\_entropy = Entropy.relative$: two implementations agree
        """
        from qudit.tools.metrics import Distance, Entropy

        rho = np.diag([0.8, 0.2]).astype(complex)
        sigma = np.diag([0.5, 0.5]).astype(complex)
        d_rel = Distance.relative_entropy(rho, sigma)

        e_rel = Entropy.relative(rho, sigma)

        self.assertAlmostEqual(
            d_rel,
            e_rel,
            places=5,
            msg="Distance.relative_entropy and Entropy.relative should agree",
        )

    def test_relative_entropy_distance_zero(self):
        """
        $D.relative\\_entropy(\\rho \\| \\rho) = 0$
        """
        from qudit.tools.metrics import Distance

        rho = np.diag([0.6, 0.4]).astype(complex)

        self.assertAlmostEqual(
            Distance.relative_entropy(rho, rho),
            0.0,
            places=4,
            msg="Relative entropy distance D(rho||rho) should be 0",
        )

    def test_relative_entropy_value(self):
        """
        $D(\\mathrm{diag}(0.8,0.2) \\| \\mathrm{diag}(0.5,0.5))
        = \\sum_i p_i \\log_2(p_i/q_i) \\approx 0.2781$ bits
        """
        from qudit.tools.metrics import Distance

        rho = np.diag([0.8, 0.2]).astype(complex)
        sigma = np.diag([0.5, 0.5]).astype(complex)

        expected = 0.8 * np.log2(0.8 / 0.5) + 0.2 * np.log2(0.2 / 0.5)

        self.assertAlmostEqual(
            Distance.relative_entropy(rho, sigma),
            expected,
            places=4,
            msg="Relative entropy distance value mismatch",
        )

    def test_trace_dist_orthogonal_pure(self):
        """
        $T(|0\\rangle\\langle 0|, |1\\rangle\\langle 1|) = 1$: maximally distinguishable pure states
        """
        from qudit.tools.metrics import Distance

        rho0 = np.array([[1, 0], [0, 0]], dtype=complex)

        rho1 = np.array([[0, 0], [0, 1]], dtype=complex)

        self.assertAlmostEqual(
            Distance.trace(rho0, rho1),
            1.0,
            places=5,
            msg="Trace distance between orthogonal pure states should be 1",
        )


class FidelityExtendedTests(Question):
    """
    Extended fidelity tests: density matrix cases, mixed states.
    """

    def test_fidelity_mixed_mixed(self):
        """
        $F(I/2, I/2) = 1$: maximally mixed state has unit fidelity with itself
        """
        from qudit.tools.metrics import Fidelity

        rho = np.eye(2, dtype=complex) / 2

        self.assertAlmostEqual(
            Fidelity.default(rho, rho), 1.0, places=5, msg="F(I/2, I/2) should be 1"
        )

    def test_fidelity_pure_density_matrix(self):
        """
        $F(|0\\rangle\\langle 0|, |1\\rangle\\langle 1|) = 0$: orthogonal density matrices
        """
        from qudit.tools.metrics import Fidelity

        rho0 = np.array([[1, 0], [0, 0]], dtype=complex)

        rho1 = np.array([[0, 0], [0, 1]], dtype=complex)

        self.assertAlmostEqual(
            Fidelity.default(rho0, rho1),
            0.0,
            places=5,
            msg="F(|0><0|, |1><1|) should be 0",
        )

    def test_fidelity_value(self):
        """
        $F(|0\\rangle, |{+}\\rangle) = |\\langle 0|{+}\\rangle|^2 = 1/2$
        """
        from qudit.tools.metrics import Fidelity

        psi = np.array([1, 0], dtype=complex)

        phi = np.array([1, 1], dtype=complex) / np.sqrt(2)

        self.assertAlmostEqual(
            Fidelity.default(psi, phi), 0.5, places=6, msg="F(|0>, |+>) should be 0.5"
        )


if __name__ == "__main__":
    runner = Exam(
        name="Entropy & Metrics Tests",
        desc="Entropy, fidelity, distance, and information measures with known values",
        file="entropy.md",
    )
    runner.run(load(EntropyTests))
    runner.run(load(FidelityTests))
    runner.run(load(DistanceTests))
    runner.run(load(InfoTests))
    runner.run(load(InfoCondtionalTests))
    runner.run(load(DistanceExtendedTests))
    runner.run(load(FidelityExtendedTests))
