import unittest
import sys

sys.path.append("..")
import numpy as np
from qudit.tools.metrics import Fidelity, Entropy, Information


class TestQuantumMetrics(unittest.TestCase):

    def test_channel(self):
        K0 = np.sqrt(0.5) * np.eye(2)
        K1 = np.sqrt(0.5) * np.array([[0, 1], [1, 0]])
        kraus_ops = [K0, K1]

        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        output = Fidelity.channel(kraus_ops, rho)

        expected = 0.5 * np.eye(2)
        np.testing.assert_almost_equal(output, expected, decimal=6)

    def test_fidelity_pure(self):
        psi = np.array([1, 0], dtype=complex)
        phi = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)
        f = Fidelity.default(psi, phi)
        self.assertAlmostEqual(f, 0.5, places=6)

    def test_negativity(self):
        d = 2
        bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        rho = np.outer(bell, bell.conj())
        N = negativity(rho, d, d)
        self.assertAlmostEqual(N, 0.5, places=6)

    def test_entropy_neumann(self):
        rho = np.array([[0.7, 0], [0, 0.3]])
        S = Entropy.neumann(rho)
        self.assertAlmostEqual(S, -0.7 * np.log2(0.7) - 0.3 * np.log2(0.3), places=6)

    def test_entropy_shannon(self):
        probs = np.array([0.5, 0.5])
        S = Entropy.shannon(probs)
        self.assertAlmostEqual(S, 1.0, places=6)

    def test_entropy_tsallis(self):
        rho = np.array([[0.6, 0], [0, 0.4]])
        S = Entropy.tsallis(rho, q=2)
        expected = (1 - (0.6**2 + 0.4**2)) / (2 - 1)
        self.assertAlmostEqual(S, expected, places=6)

    def test_entropy_renyi(self):
        rho = np.array([[0.6, 0], [0, 0.4]])
        S = Entropy.renyi(rho, alpha=2)
        expected = np.log2(0.6**2 + 0.4**2) / (1 - 2)
        self.assertAlmostEqual(S, expected, places=6)

    def test_entropy_hartley(self):
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        S = Entropy.hartley(probs)
        self.assertAlmostEqual(S, 2.0, places=6)

    def test_entropy_unified(self):
        rho = np.array([[0.6, 0], [0, 0.4]])
        S1 = Entropy.unified(rho, q=1.0, alpha=2.0)
        S2 = Entropy.unified(rho, q=2.0, alpha=1.0)
        S3 = Entropy.unified(rho, q=2.0, alpha=2.0)
        self.assertAlmostEqual(S1, Entropy.renyi(rho, alpha=2.0), places=6)
        self.assertAlmostEqual(S2, Entropy.tsallis(rho, q=2.0), places=6)
        self.assertTrue(S3 > 0)

    def test_relative_entropy(self):
        rho = np.array([[0.8, 0], [0, 0.2]])
        sigma = np.eye(2) / 2
        D = Entropy.relative_entropy(rho, sigma)
        expected = 0.8 * np.log2(0.8 / 0.5) + 0.2 * np.log2(0.2 / 0.5)
        self.assertAlmostEqual(D, expected, places=6)

    def test_mutual_information_bell_state():
        psi = np.array([1, 0, 0, 1]) / np.sqrt(2)
        rho = np.outer(psi, psi.conj())
        I = Information.mutual_information(rho, 2, 2)
        assert np.isclose(I, 2.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
