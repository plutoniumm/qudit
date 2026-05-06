from MDR import Exam, load, Question
import sys

sys.path.append("..")

from qudit.algo import QAOA
from qudit.algo.qaoa import QUBO, ClockSolver
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

        self.assertAlmostEqual(
            offset, 1.0, places=6, msg="QUBO diagonal term offset should be 1.0"
        )

        self.assertEqual(
            len(ham),
            1,
            msg="QUBO diagonal single var should produce 1 Hamiltonian term",
        )
        coeff, gtype, indices = ham[0]

        self.assertAlmostEqual(
            coeff, -1.0, places=6, msg="QUBO diagonal term coefficient should be -1.0"
        )

        self.assertEqual(gtype, "Z", msg="QUBO diagonal term should be a Z gate")

        self.assertEqual(indices, [0], msg="QUBO diagonal term should act on qubit 0")

    def test_quadratic_term(self):
        """
        $Q_{01} = 4 \\Rightarrow ZZ + Z_0 + Z_1 + \\mathrm{const}$
        with coefficients $1, -1, -1$ and offset $1$
        """
        ham, offset = QUBO.toHamiltonian({(0, 1): 4.0})

        self.assertAlmostEqual(
            offset, 1.0, places=6, msg="QUBO quadratic term offset should be 1.0"
        )
        terms = {(gtype, tuple(idx)): coeff for coeff, gtype, idx in ham}

        self.assertAlmostEqual(
            terms[("ZZ", (0, 1))],
            1.0,
            places=6,
            msg="ZZ term coefficient should be 1.0",
        )

        self.assertAlmostEqual(
            terms[("Z", (0,))], -1.0, places=6, msg="Z0 term coefficient should be -1.0"
        )

        self.assertAlmostEqual(
            terms[("Z", (1,))], -1.0, places=6, msg="Z1 term coefficient should be -1.0"
        )

    def test_energy_consistency(self):
        """
        QUBO energy at $x=[1,0]$ equals Ising expectation $(Z_0=-1, Z_1=+1)$ plus offset,
        two ways: direct sum vs. Ising evaluation
        """
        Q = {
            (0, 0): -2.0,
            (0, 1): 1.0,
            (1, 1): -2.0,
        }
        ham, offset = QUBO.toHamiltonian(Q)
        qubo_energy = -2.0
        z_vals = {
            0: -1,
            1: 1,
        }
        ising_energy = offset
        for coeff, gtype, indices in ham:
            if gtype == "Z":
                ising_energy += coeff * z_vals[indices[0]]
            elif gtype == "ZZ":
                ising_energy += coeff * z_vals[indices[0]] * z_vals[indices[1]]

        self.assertAlmostEqual(
            float(ising_energy),
            qubo_energy,
            places=5,
            msg="QUBO and Ising energies should match",
        )

    def test_zero_offset_for_linear(self):
        """
        $Q = \\{(0,0): -3\\} \\Rightarrow \\mathrm{offset} = -3/2$,
        symmetric about the $x=0, x=1$ energies
        """
        ham, offset = QUBO.toHamiltonian({(0, 0): -3.0})

        self.assertAlmostEqual(
            offset, -1.5, places=6, msg="Negative diagonal QUBO offset should be -1.5"
        )
        coeff, _, _ = ham[0]

        self.assertAlmostEqual(
            coeff,
            1.5,
            places=6,
            msg="Negative diagonal QUBO Z coefficient should be 1.5",
        )


class QAOACircuit(Question):
    """
    QAOA circuit tests: normalized state, real expectation, eigenvalue bounds,
    and optimization descent.
    """

    def _qaoa(self, wires=2, layers=1):
        Q = {
            (0, 0): -1.0,
            (1, 1): -1.0,
        }

        return QAOA(d=2, wires=wires, qubo=Q, layers=layers, device="cpu")

    def test_forward_normalized(self):
        """
        QAOA forward pass produces a normalized state: $\\|\\psi\\|_2 = 1$
        """

        state = self._qaoa().forward()

        self.assertAlmostEqual(
            pt.norm(state).item(),
            1.0,
            places=5,
            msg="QAOA forward pass should produce normalized state",
        )

    def test_expectation_real(self):
        """
        Expectation value $\\langle H \\rangle \\in \\mathbb{R}$
        """

        exp = self._qaoa().expectation()

        self.assertIsInstance(
            float(exp), float, msg="QAOA expectation value should be a real float"
        )

    def test_expectation_bounded_by_eigenvalues(self):
        """
        $\\lambda_{\\min}(H_P) \\leq \\langle H_P \\rangle \\leq \\lambda_{\\max}(H_P)$:
        expectation of the problem Hamiltonian lies within its spectral range
        """
        qaoa = self._qaoa()
        H = qaoa._H_P_matrix
        eigs = pt.linalg.eigvalsh(H.real).numpy()

        exp_val = qaoa.expectation().item() - qaoa.offset

        self.assertGreaterEqual(
            exp_val,
            float(eigs.min()) - 1e-5,
            msg="QAOA expectation should be >= min eigenvalue",
        )

        self.assertLessEqual(
            exp_val,
            float(eigs.max()) + 1e-5,
            msg="QAOA expectation should be <= max eigenvalue",
        )

    def test_hamiltonian_hermitian(self):
        """
        $H_P = H_P^\\dagger$: the problem Hamiltonian is Hermitian
        """
        H = self._qaoa()._H_P_matrix

        diff = pt.norm(H - H.conj().T).item()

        self.assertAlmostEqual(
            diff, 0.0, delta=1e-5, msg="QAOA Hamiltonian should be Hermitian"
        )

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

        self.assertLessEqual(
            final,
            initial + 1e-3,
            msg="QAOA optimization should not increase expectation value",
        )


class QAOASolve(Question):
    """
    QAOA.solve() output contract and optimization tests.
    """

    Q = {
        (0, 0): -1.0,
        (1, 1): -1.0,
        (0, 1): 2.0,
    }

    def _qaoa(self, d=2, layers=1):
        pt.manual_seed(42)

        return QAOA(d=d, wires=2, qubo=self.Q, layers=layers, device="cpu")

    def test_solve_returns_dict(self):
        """
        solve() returns a dict containing keys "solution" and "probabilities"
        """
        out = self._qaoa().solve(steps=30, lr=0.1)

        self.assertIn("solution", out, msg="solve() output missing 'solution' key")

        self.assertIn(
            "probabilities", out, msg="solve() output missing 'probabilities' key"
        )

    def test_solution_is_int_list(self):
        """
        solve()["solution"] is a list of ints, one per wire
        """
        sol = self._qaoa().solve(steps=30, lr=0.1)["solution"]

        self.assertIsInstance(sol, list, msg="solution should be a list")

        self.assertEqual(
            len(sol), 2, msg="solution length should equal number of wires"
        )

        for bit in sol:
            self.assertIsInstance(
                bit, int, msg=f"each solution element should be int, got {type(bit)}"
            )

    def test_probabilities_sum_to_one(self):
        """
        solve()["probabilities"] sums to $1$ (normalized distribution over bitstrings)
        """
        probs = self._qaoa().solve(steps=30, lr=0.1)["probabilities"]
        total = float(probs.sum().item())

        self.assertAlmostEqual(
            total, 1.0, places=5, msg="probabilities should sum to 1.0"
        )

    def test_probabilities_length(self):
        """
        solve()["probabilities"] has $d^n = 4$ entries for 2 qubits
        """
        probs = self._qaoa().solve(steps=30, lr=0.1)["probabilities"]

        self.assertEqual(
            len(probs.flatten()),
            4,
            msg="probabilities should have d^wires entries",
        )

    def test_two_layers_runs(self):
        """
        QAOA with layers=2 runs solve() without error
        """
        pt.manual_seed(42)
        qaoa = QAOA(d=2, wires=2, qubo=self.Q, layers=2, device="cpu")
        out = qaoa.solve(steps=20, lr=0.1)

        self.assertIn("solution", out, msg="layers=2 solve() should return solution")

    def test_qutrit_runs(self):
        """
        QAOA with $d=3$ (qutrit) runs solve() without error
        """
        pt.manual_seed(42)
        qaoa = QAOA(d=3, wires=2, qubo=self.Q, layers=1, device="cpu")
        out = qaoa.solve(steps=20, lr=0.1)

        self.assertIn("solution", out, msg="d=3 QAOA solve() should return solution")

        self.assertEqual(
            len(out["solution"]),
            2,
            msg="d=3 solution length should equal number of wires",
        )


class ClockSolverTests(Question):
    """
    ClockSolver variational qudit optimizer tests for $d=2$ and $d=3$.
    """

    Q = {
        (0, 0): -1.0,
        (1, 1): -1.0,
        (0, 1): 2.0,
    }

    def _solver(self, d=2, layers=1):
        pt.manual_seed(42)

        return ClockSolver(d=d, wires=2, qubo=self.Q, layers=layers, device="cpu")

    def test_d2_solve_returns_dict(self):
        """
        ClockSolver($d=2$) solve() returns dict with "solution" and "probabilities"
        """
        out = self._solver(d=2).solve(steps=30, lr=0.1)

        self.assertIn("solution", out, msg="d=2 solve() missing 'solution'")

        self.assertIn("probabilities", out, msg="d=2 solve() missing 'probabilities'")

    def test_d2_solution_is_int_list(self):
        """
        ClockSolver($d=2$) solution is a list of 2 ints
        """
        sol = self._solver(d=2).solve(steps=30, lr=0.1)["solution"]

        self.assertIsInstance(sol, list, msg="d=2 solution should be a list")

        self.assertEqual(len(sol), 2, msg="d=2 solution length should be 2")

        for v in sol:
            self.assertIsInstance(
                v, int, msg=f"d=2 solution element should be int, got {type(v)}"
            )

    def test_d3_solve_returns_dict(self):
        """
        ClockSolver($d=3$) solve() returns dict with "solution" and "probabilities"
        """
        out = self._solver(d=3).solve(steps=30, lr=0.1)

        self.assertIn("solution", out, msg="d=3 solve() missing 'solution'")

        self.assertIn("probabilities", out, msg="d=3 solve() missing 'probabilities'")

    def test_d3_solution_values_in_range(self):
        """
        ClockSolver($d=3$) solution elements are in $\\{0, 1, 2\\}$
        """
        sol = self._solver(d=3).solve(steps=30, lr=0.1)["solution"]

        for v in sol:
            self.assertIn(
                v,
                [0, 1, 2],
                msg=f"d=3 solution element {v} not in {{0,1,2}}",
            )

    def test_expectation_is_scalar_tensor(self):
        """
        expectation() returns a scalar torch.Tensor
        """
        exp = self._solver().expectation()

        self.assertIsInstance(
            exp, pt.Tensor, msg="expectation() should return a torch.Tensor"
        )

        self.assertEqual(
            exp.shape,
            pt.Size([]),
            msg="expectation() should be a scalar (0-dim tensor)",
        )

    def test_forward_normalized(self):
        """
        forward() produces a state with $\\|\\psi\\|_2 \\approx 1$
        """
        state = self._solver().forward()
        norm = pt.norm(state).item()

        self.assertAlmostEqual(
            norm, 1.0, places=5, msg="ClockSolver forward() state should be normalized"
        )

    def test_d3_forward_normalized(self):
        """
        ClockSolver($d=3$) forward() state is also normalized
        """
        state = self._solver(d=3).forward()
        norm = pt.norm(state).item()

        self.assertAlmostEqual(
            norm,
            1.0,
            places=5,
            msg="ClockSolver d=3 forward() state should be normalized",
        )


class QUBOEdgeCases(Question):
    """
    Edge-case and symmetry tests for QUBO.toHamiltonian().
    """

    def test_empty_q(self):
        """
        Empty $Q$ gives empty Hamiltonian and zero offset
        """
        ham, offset = QUBO.toHamiltonian({})

        self.assertEqual(len(ham), 0, msg="empty Q should give empty Hamiltonian")

        self.assertAlmostEqual(
            offset, 0.0, places=6, msg="empty Q should give zero offset"
        )

    def test_multiple_diagonal_terms(self):
        """
        Three diagonal entries produce exactly 3 $Z$ terms (one per variable)
        """
        Q = {
            (0, 0): 1.0,
            (1, 1): 2.0,
            (2, 2): 3.0,
        }
        ham, _ = QUBO.toHamiltonian(Q)

        z_terms = [(c, gt, idx) for c, gt, idx in ham if gt == "Z"]

        self.assertEqual(
            len(z_terms), 3, msg="three diagonal entries should produce 3 Z terms"
        )

    def test_diagonal_offset_sum(self):
        """
        Offset equals $\sum_i Q_{ii}/2$ for all-diagonal $Q$
        """
        Q = {
            (0, 0): 2.0,
            (1, 1): 4.0,
            (2, 2): 6.0,
        }
        _, offset = QUBO.toHamiltonian(Q)

        expected = (2.0 + 4.0 + 6.0) / 2

        self.assertAlmostEqual(
            offset,
            expected,
            places=6,
            msg=f"all-diagonal offset should be {expected}",
        )

    def test_symmetry_ij_vs_ji(self):
        """
        $Q = \\{(0,1): v\\}$ and $Q = \\{(1,0): v\\}$ produce identical Hamiltonians
        (QUBO is symmetric in off-diagonal indices)
        """
        v = 3.0
        ham_ij, off_ij = QUBO.toHamiltonian({(0, 1): v})
        ham_ji, off_ji = QUBO.toHamiltonian({(1, 0): v})

        self.assertAlmostEqual(
            off_ij, off_ji, places=6, msg="offsets should match for (0,1) vs (1,0)"
        )

        terms_ij = {(gt, tuple(idx)): c for c, gt, idx in ham_ij}
        terms_ji = {(gt, tuple(idx)): c for c, gt, idx in ham_ji}

        self.assertEqual(
            terms_ij.keys(),
            terms_ji.keys(),
            msg="Hamiltonian term keys should match for (0,1) vs (1,0)",
        )

        for key in terms_ij:
            self.assertAlmostEqual(
                terms_ij[key],
                terms_ji[key],
                places=6,
                msg=f"coefficient for {key} should match for (0,1) vs (1,0)",
            )


if __name__ == "__main__":
    runner = Exam(
        name="Qudit QUBO Tests",
        desc="Validation of QUBO-to-Hamiltonian conversion and QAOA circuit",
        file="qubo.md",
    )
    runner.run(load(QUBOConversion))
    runner.run(load(QAOACircuit))
    runner.run(load(QAOASolve))
    runner.run(load(ClockSolverTests))
    runner.run(load(QUBOEdgeCases))
