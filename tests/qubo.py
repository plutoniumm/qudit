from MDR import Exam, load, Question
import sys

sys.path.append("..")

from qudit.algo import QAOA
from qudit.algo.qaoa import QUBO
import torch as pt
import numpy as np


class QUBOConversion(Question):
    """
    QUBO-to-Ising Hamiltonian conversion tests for $\\mathrm{toHamiltonian}()$.
    """

    def test_diagonal_single_var(self):
        """
        $Q_{00} = 2 \\Rightarrow H = -Z_0 + 1$:
        diagonal term maps to single Pauli $Z$ with coefficient $-Q_{00}/2$
        """
        ham, offset = QUBO.toHamiltonian({(0, 0): 2.0})
        self.assertAlmostEqual(offset, 1.0, places=6)
        self.assertEqual(len(ham), 1)
        coeff, gtype, indices = ham[0]
        self.assertAlmostEqual(coeff, -1.0, places=6)
        self.assertEqual(gtype, "Z")
        self.assertEqual(indices, [0])

    def test_quadratic_term(self):
        """
        $Q_{01} = 4 \\Rightarrow ZZ + Z_0 + Z_1 + \\mathrm{const}$
        with coefficients $1, -1, -1$ and offset $1$
        """
        ham, offset = QUBO.toHamiltonian({(0, 1): 4.0})
        self.assertAlmostEqual(offset, 1.0, places=6)
        terms = {(gtype, tuple(idx)): coeff for coeff, gtype, idx in ham}
        self.assertAlmostEqual(terms[("ZZ", (0, 1))], 1.0, places=6)
        self.assertAlmostEqual(terms[("Z", (0,))], -1.0, places=6)
        self.assertAlmostEqual(terms[("Z", (1,))], -1.0, places=6)

    def test_energy_consistency(self):
        """
        QUBO energy at $x=[1,0]$ equals Ising expectation $(Z_0=-1, Z_1=+1)$ plus offset,
        two ways: direct sum vs. Ising evaluation
        """
        Q = {(0, 0): -2.0, (0, 1): 1.0, (1, 1): -2.0}
        ham, offset = QUBO.toHamiltonian(Q)
        # Direct QUBO energy at x=[1,0]: -2*1 + 0 + 0 = -2
        qubo_energy = -2.0
        # Ising: x=1 maps to Z=-1, x=0 maps to Z=+1
        z_vals = {0: -1, 1: 1}
        ising_energy = offset
        for coeff, gtype, indices in ham:
            if gtype == "Z":
                ising_energy += coeff * z_vals[indices[0]]
            elif gtype == "ZZ":
                ising_energy += coeff * z_vals[indices[0]] * z_vals[indices[1]]
        self.assertAlmostEqual(float(ising_energy), qubo_energy, places=5)

    def test_zero_offset_for_linear(self):
        """
        $Q = \\{(0,0): -3\\} \\Rightarrow \\mathrm{offset} = -3/2$,
        symmetric about the $x=0, x=1$ energies
        """
        ham, offset = QUBO.toHamiltonian({(0, 0): -3.0})
        self.assertAlmostEqual(offset, -1.5, places=6)
        coeff, _, _ = ham[0]
        self.assertAlmostEqual(coeff, 1.5, places=6)


class QAOACircuit(Question):
    """
    QAOA circuit tests: normalized state, real expectation, eigenvalue bounds,
    and optimization descent.
    """

    def _qaoa(self, wires=2, layers=1):
        Q = {(0, 0): -1.0, (1, 1): -1.0}
        return QAOA(d=2, wires=wires, qubo=Q, layers=layers, device="cpu")

    def test_forward_normalized(self):
        """
        QAOA forward pass produces a normalized state: $\\|\\psi\\|_2 = 1$
        """
        state = self._qaoa().forward()
        self.assertAlmostEqual(pt.norm(state).item(), 1.0, places=5)

    def test_expectation_real(self):
        """
        Expectation value $\\langle H \\rangle \\in \\mathbb{R}$
        """
        exp = self._qaoa().expectation()
        self.assertIsInstance(float(exp), float)

    def test_expectation_bounded_by_eigenvalues(self):
        """
        $\\lambda_{\\min}(H_P) \\leq \\langle H_P \\rangle \\leq \\lambda_{\\max}(H_P)$:
        expectation of the problem Hamiltonian lies within its spectral range
        """
        qaoa = self._qaoa()
        H = qaoa._H_P_matrix
        eigs = pt.linalg.eigvalsh(H.real).numpy()
        exp_val = qaoa.expectation().item() - qaoa.offset
        self.assertGreaterEqual(exp_val, float(eigs.min()) - 1e-5)
        self.assertLessEqual(exp_val, float(eigs.max()) + 1e-5)

    def test_hamiltonian_hermitian(self):
        """
        $H_P = H_P^\\dagger$: the problem Hamiltonian is Hermitian
        """
        H = self._qaoa()._H_P_matrix
        diff = pt.norm(H - H.conj().T).item()
        self.assertAlmostEqual(diff, 0.0, delta=1e-5)

    def test_optimization_decreases_loss(self):
        """
        After 20 Adam steps, QAOA expectation does not increase:
        $\\langle H \\rangle_{\\mathrm{final}} \\leq \\langle H \\rangle_{\\mathrm{initial}} + \\epsilon$
        """
        qaoa = self._qaoa(layers=2)
        initial = qaoa.expectation().item()
        optimizer = pt.optim.Adam(qaoa.parameters(), lr=0.1)
        for _ in range(20):
            optimizer.zero_grad()
            qaoa.expectation().backward()
            optimizer.step()
        final = qaoa.expectation().item()
        self.assertLessEqual(final, initial + 1e-3)


if __name__ == "__main__":
    runner = Exam(
        name="Qudit QUBO Tests",
        desc="Validation of QUBO-to-Hamiltonian conversion and QAOA circuit",
        file="qubo.md",
    )
    runner.run(load(QUBOConversion))
    runner.run(load(QAOACircuit))
