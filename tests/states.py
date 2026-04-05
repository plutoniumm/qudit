from MDR import Exam, load, Question
import sys
import numpy as np
import torch as pt

sys.path.append("..")

from qudit.tools.states import GHZ, W, NOON, Dicke, Coherent
from qudit.tools.metrics import Entropy
from qudit import Basis, State


class SpecialStates(Question):
    """
    State construction: norm, amplitudes, and entropy for named quantum states.
    """

    def test_ghz_norm(self):
        """
        GHZ(2,2) is normalized
        """
        s = GHZ(2, 2)

        self.assertAlmostEqual(np.linalg.norm(s), 1.0, places=6, msg="GHZ(2,2) should be normalized")

    def test_ghz_structure(self):
        """
        GHZ(2,2) = (|00⟩+|11⟩)/√2: amplitudes at indices 0 and 3
        """
        s = GHZ(2, 2)
        arr = np.array(s)

        self.assertAlmostEqual(float(np.abs(arr[0])), 1 / np.sqrt(2), places=6, msg="GHZ(2,2) amplitude at |00> should be 1/√2")

        self.assertAlmostEqual(float(np.abs(arr[3])), 1 / np.sqrt(2), places=6, msg="GHZ(2,2) amplitude at |11> should be 1/√2")

        self.assertAlmostEqual(float(np.abs(arr[1])), 0.0, places=6, msg="GHZ(2,2) amplitude at |01> should be 0")

        self.assertAlmostEqual(float(np.abs(arr[2])), 0.0, places=6, msg="GHZ(2,2) amplitude at |10> should be 0")

    def test_ghz_reduced_entropy(self):
        """
        GHZ(2,2) reduced state is maximally mixed: entropy = 1 ebit
        """
        s = GHZ(2, 2)
        rho = np.outer(s, s.conj())
        rho_A = np.trace(rho.reshape(2, 2, 2, 2), axis1=1, axis2=3)
        ent = Entropy.neumann(rho_A)

        self.assertAlmostEqual(ent, 1.0, places=5, msg="GHZ(2,2) reduced state should have entropy = 1 ebit")

    def test_w_norm(self):
        """
        W(3) is normalized
        """
        s = W(3)

        self.assertAlmostEqual(np.linalg.norm(s), 1.0, places=6, msg="W(3) should be normalized")

    def test_w_equal_amplitudes(self):
        """
        W(3) has three equal-magnitude amplitudes of 1/√3
        """
        s = W(3)
        arr = np.array(s)
        nonzero = np.abs(arr[arr != 0])

        self.assertEqual(len(nonzero), 3, msg="W(3) should have 3 nonzero amplitudes")
        for v in nonzero:
            self.assertAlmostEqual(float(v), 1 / np.sqrt(3), places=6, msg="W(3) amplitudes should be 1/√3")

    def test_noon_norm(self):
        """
        NOON(2, theta=0) is normalized
        """
        s = NOON(2, theta=0.0)

        self.assertAlmostEqual(np.linalg.norm(s), 1.0, places=6, msg="NOON(2, theta=0) should be normalized")

    def test_noon_two_terms(self):
        """
        NOON(2) has exactly two non-zero amplitudes of equal magnitude
        """
        s = NOON(2, theta=0.0)
        arr = np.array(s)
        nonzero = np.abs(arr[np.abs(arr) > 1e-10])

        self.assertEqual(len(nonzero), 2, msg="NOON(2) should have 2 nonzero amplitudes")

        self.assertAlmostEqual(float(nonzero[0]), 1 / np.sqrt(2), places=6, msg="NOON(2) first amplitude should be 1/√2")

        self.assertAlmostEqual(float(nonzero[1]), 1 / np.sqrt(2), places=6, msg="NOON(2) second amplitude should be 1/√2")

    def test_dicke_norm(self):
        """
        Dicke(4,2) is normalized
        """
        s = Dicke(4, 2)

        self.assertAlmostEqual(np.linalg.norm(s), 1.0, places=6, msg="Dicke(4,2) should be normalized")

    def test_dicke_term_count(self):
        """
        Dicke(4,2) has C(4,2)=6 non-zero amplitudes each of magnitude 1/√6
        """
        s = Dicke(4, 2)
        arr = np.array(s)
        nonzero = np.abs(arr[np.abs(arr) > 1e-10])

        self.assertEqual(len(nonzero), 6, msg="Dicke(4,2) should have C(4,2)=6 nonzero amplitudes")
        for v in nonzero:
            self.assertAlmostEqual(float(v), 1 / np.sqrt(6), places=6, msg="Dicke(4,2) amplitudes should be 1/√6")

    def test_coherent_norm(self):
        """
        Coherent(5, alpha=1.0) is approximately normalized
        """
        s = Coherent(5, alpha=1.0)

        self.assertAlmostEqual(np.linalg.norm(s), 1.0, places=4, msg="Coherent(5, alpha=1.0) should be approximately normalized")

    def test_ghz_qutrit(self):
        """
        GHZ(2,3) has amplitudes at |00⟩,|11⟩,|22⟩ of 1/√3
        """
        s = GHZ(2, 3)
        arr = np.array(s)
        # In d=3, n=2: |00⟩=0, |11⟩=4, |22⟩=8
        self.assertAlmostEqual(float(np.abs(arr[0])), 1 / np.sqrt(3), places=6, msg="GHZ(2,3) amplitude at |00> should be 1/√3")

        self.assertAlmostEqual(float(np.abs(arr[4])), 1 / np.sqrt(3), places=6, msg="GHZ(2,3) amplitude at |11> should be 1/√3")

        self.assertAlmostEqual(float(np.abs(arr[8])), 1 / np.sqrt(3), places=6, msg="GHZ(2,3) amplitude at |22> should be 1/√3")


class QuditStates(Question):
    """
    W and Dicke states for d > 2: norm, shape, and term count.
    """

    def test_w_qutrit_norm(self):
        """
        $W(3, d=3)$ is normalized and lives in $3^3=27$-dimensional space
        """
        s = W(3, d=3)

        self.assertAlmostEqual(float(np.linalg.norm(s)), 1.0, places=6, msg="W(3, d=3) should be normalized")

        self.assertEqual(len(s), 27, msg="W(3, d=3) should live in 3^3=27-dimensional space")

    def test_w_qutrit_term_count(self):
        """
        $W(3, d=3)$ has exactly 3 equal-magnitude terms of $1/\\sqrt{3}$
        """
        s = W(3, d=3)
        arr = np.array(s)
        nonzero = np.abs(arr[np.abs(arr) > 1e-10])

        self.assertEqual(len(nonzero), 3, msg="W(3, d=3) should have 3 nonzero amplitudes")
        for v in nonzero:
            self.assertAlmostEqual(float(v), 1 / np.sqrt(3), places=6, msg="W(3, d=3) amplitudes should be 1/√3")

    def test_w_qutrit_d2_unchanged(self):
        """
        $W(3, d=2)$ equals $W(3)$: d=2 default is unchanged
        """
        self.stateEqual(np.array(W(3)), np.array(W(3, d=2)), msg="W(3, d=2) should equal W(3)")

    def test_dicke_qutrit_norm(self):
        """
        $Dicke(3, 1, d=3)$ is normalized and lives in $3^3=27$-dimensional space
        """
        s = Dicke(3, 1, d=3)

        self.assertAlmostEqual(float(np.linalg.norm(s)), 1.0, places=6, msg="Dicke(3, 1, d=3) should be normalized")

        self.assertEqual(len(s), 27, msg="Dicke(3, 1, d=3) should live in 3^3=27-dimensional space")

    def test_dicke_qutrit_d2_unchanged(self):
        """
        $Dicke(4, 2, d=2)$ equals $Dicke(4, 2)$: d=2 default is unchanged
        """
        self.stateEqual(np.array(Dicke(4, 2)), np.array(Dicke(4, 2, d=2)), msg="Dicke(4,2,d=2) should equal Dicke(4,2)")


class WStatePhysics(Question):
    """
    W-state physics: Dicke relationship, Schmidt rank, structure for various n.
    """

    def test_dicke_n3_k2_equals_w3(self):
        """
        $Dicke(3, 2) = W(3)$: Dicke state with $k=2$ zeros and 1 one equals $W(3)$
        """
        d = Dicke(3, 2)
        w = W(3)

        self.stateEqual(np.array(d), np.array(w), msg="Dicke(3,2) should equal W(3)")

    def test_w_varies_n(self):
        """
        $W(n)$ normalized for $n = 2, 3, 4, 5$: general W state norm check
        """
        for n in [2, 3, 4, 5]:
            w = W(n)
            norm = float(np.linalg.norm(w))

            self.assertAlmostEqual(norm, 1.0, places=6, msg=f"W({n}) not normalized")

    def test_w4_nonzero_count(self):
        """
        $W(4)$ has exactly 4 equal-amplitude terms of $1/2$
        """
        w = W(4)
        arr = np.array(w)
        nonzero = np.abs(arr[np.abs(arr) > 1e-10])

        self.assertEqual(len(nonzero), 4, msg="W(4) should have 4 nonzero amplitudes")
        for v in nonzero:
            self.assertAlmostEqual(
                float(v), 0.5, places=6, msg="W(4) amplitudes should be 1/2"
            )


class NOONStatePhysics(Question):
    """
    NOON state physics: entanglement, phase dependence, dimension checks.
    """

    def test_noon_phase_dependence(self):
        """
        $NOON(2, \\pi/2)$: phase $e^{i \\pi} = -1$ on $|02\\rangle$ component,
        so relative phase between $|20\\rangle$ and $|02\\rangle$ is $e^{2i\\pi/2}=e^{i\\pi}=-1$
        """
        from qudit import Basis

        Ket = Basis(3)
        noon = NOON(2, np.pi / 2)
        expected = State(Ket(2, 0) + np.exp(1j * 2 * np.pi / 2) * Ket(0, 2))

        self.stateEqual(
            np.array(noon), np.array(expected), msg="NOON(2, pi/2) phase mismatch"
        )

    def test_noon_dimension(self):
        """
        $NOON(N, 0)$ lives in $(N+1)^2$-dimensional space
        """
        for N in [1, 2, 3]:
            noon = NOON(N, 0.0)

            self.assertEqual(len(noon), (N + 1) ** 2, msg=f"NOON({N}) wrong dimension")

    def test_noon3_structure(self):
        """
        $NOON(3, 0) = (|30\\rangle + |03\\rangle)/\\sqrt{2}$ in $d=4$ local space
        """
        from qudit import Basis

        Ket = Basis(4)
        noon = NOON(3, 0.0)
        expected = State(Ket(3, 0) + Ket(0, 3))

        self.stateEqual(
            np.array(noon), np.array(expected), msg="NOON(3,0) structure mismatch"
        )


class CoherentStatePhysics(Question):
    """
    Coherent state physics: vacuum, Poisson statistics, varying alpha.
    """

    def test_coherent_mean_photon_number(self):
        """
        $\\langle n \\rangle = \\sum_n n |c_n|^2 \\approx |\\alpha|^2$ for large $N$:
        mean photon number of coherent state equals $|\\alpha|^2$
        """
        N, alpha = 15, 2.0
        c = Coherent(N, alpha)
        arr = np.array(c)
        probs = np.abs(arr) ** 2
        mean_n = float(np.sum(np.arange(N + 1) * probs))
        # For truncation N=15, alpha=2: mean should be very close to |alpha|^2 = 4
        self.assertAlmostEqual(
            mean_n,
            alpha**2,
            places=2,
            msg=f"Coherent state mean photon number should be |alpha|^2",
        )

    def test_coherent_varying_alpha(self):
        """
        $Coherent(8, \\alpha)$ normalized for various complex $\\alpha$
        """
        for alpha in [0.5, 1.0, 1.5, 1j, 0.5 + 0.5j]:
            c = Coherent(8, alpha)
            norm = float(np.linalg.norm(c))

            self.assertAlmostEqual(
                norm, 1.0, places=4, msg=f"Coherent(8, {alpha}) not normalized"
            )


class DickeStatePhysics(Question):
    """
    Dicke state physics: dimension, term count for various (n,k).
    """

    def test_dicke_31_term_count(self):
        """
        $Dicke(3,1)$ has $C(3,1)=3$ non-zero terms (2 ones, 1 zero)
        """
        d = Dicke(3, 1)
        arr = np.array(d)
        nonzero = int(np.count_nonzero(np.abs(arr) > 1e-10))

        self.assertEqual(nonzero, 3, msg="Dicke(3,1) should have 3 terms")

    def test_dicke_53_term_count(self):
        """
        $Dicke(5,3)$ has $C(5,3)=10$ non-zero terms
        """
        d = Dicke(5, 3)
        arr = np.array(d)
        nonzero = int(np.count_nonzero(np.abs(arr) > 1e-10))

        self.assertEqual(nonzero, 10, msg="Dicke(5,3) should have 10 terms")

    def test_dicke_amplitude_formula(self):
        """
        $Dicke(n,k)$ amplitudes are all $1/\\sqrt{C(n,k)}$
        """
        import math

        for n, k in [(4, 2), (4, 1), (5, 2), (3, 1)]:
            d = Dicke(n, k)
            arr = np.array(d)
            expected_amp = 1.0 / np.sqrt(math.comb(n, k))
            nonzero_amps = np.abs(arr[np.abs(arr) > 1e-10])
            for a in nonzero_amps:
                self.assertAlmostEqual(
                    float(a),
                    expected_amp,
                    places=5,
                    msg=f"Dicke({n},{k}) amplitude mismatch",
                )


if __name__ == "__main__":
    runner = Exam(
        name="Special State Tests",
        desc="Norm, structure, and entropy of GHZ, W, NOON, Dicke, Coherent states",
        file="states.md",
    )
    runner.run(load(SpecialStates))
    runner.run(load(QuditStates))
    runner.run(load(WStatePhysics))
    runner.run(load(NOONStatePhysics))
    runner.run(load(CoherentStatePhysics))
    runner.run(load(DickeStatePhysics))
