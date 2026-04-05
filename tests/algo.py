from MDR import Exam, load, Question
import sys

sys.path.append("..")

from qudit.tools.entanglement import Loss, rank, Perp
from qudit.algo import Statiliser
from qudit.algo.statiliser import S
from qudit import Basis, State
import numpy as np
import torch as pt


class StabiliserCode(Question):
    """
    Stabilizer subspace tests: codeword count, orthonormality, and eigenstate condition.
    """

    STABS = ["ZZZII", "IIZZZ", "XIXXI", "IXXIX"]

    def test_num_states(self):
        """
        $2^{n-k}$ stabilized codewords: $n=5$ qubits, $k=4$ generators $\\Rightarrow 2$ codewords
        """
        s = Statiliser(self.STABS)

        self.assertEqual(
            s.num_states, 2, msg="Expected 2 stabilizer codewords for n=5, k=4"
        )

    def test_states_orthonormal(self):
        """
        Generated codewords form an orthonormal basis:
        $\\langle \\psi_i | \\psi_j \\rangle = \\delta_{ij}$, verified via Gram matrix
        """
        pt.manual_seed(0)
        s = Statiliser(self.STABS)
        basis = s.generate(minimal=False)
        gram = pt.real(basis @ basis.conj().T).numpy()

        np.testing.assert_allclose(gram, np.eye(2), atol=1e-3)

    def test_stabilizer_eigenstate(self):
        """
        Each codeword is a $+1$ eigenstate of every generator (using exact mode):
        $g|\\psi_i\\rangle = |\\psi_i\\rangle$ for all $g \\in \\mathcal{S}$
        """
        pt.manual_seed(0)
        s = Statiliser(self.STABS)
        basis = s.generate(minimal=False)
        for state in basis:
            state_c = state.to(pt.complex64)
            for stab_str in self.STABS:
                G = S(stab_str).to(pt.complex64)
                diff = pt.norm(G @ state_c - state_c).item()

                self.assertAlmostEqual(
                    diff,
                    0.0,
                    delta=1e-2,
                    msg=f"State not +1 eigenstate of {stab_str}",
                )

    def test_qutrit_num_states(self):
        """
        Qutrit ($d=3$) stabilizer code: $d^{n-k}=3^{2-2}=1$ codeword, $\\mathrm{sz}=2$
        """
        s = Statiliser(["ZI", "IZ"], d=3)

        self.assertEqual(
            s.num_states, 1, msg="Qutrit d=3 stabilizer code should have 1 codeword"
        )

        self.assertEqual(s.sz, 2, msg="Qutrit stabilizer code should have sz=2")


class EntanglementRank(Question):
    """
    Entanglement rank tests via stabilizer-generated qubit-qutrit subspaces.
    """

    Bits = Basis(2)
    Trits = Basis(3)
    THETA = 0.75
    D = 5
    r = 2

    def Psi(self, i):
        A = self.Bits(0) ^ self.Trits(i)
        B = self.Bits(1) ^ self.Trits(i + 1)

        return A * np.cos(self.THETA) + B * np.sin(self.THETA)

    def system(self, X):
        toCplx = np.array([1, 1j])
        qbit = State(X[1:5].reshape(2, 2).dot(toCplx))
        qtrit = State(X[5:11].reshape(3, 2).dot(toCplx))
        phi_rx = (X[0] * (qbit ^ qtrit)).norm()

        return Loss(phi_rx, self.perp)

    def test_rank(self):
        """
        $\\mathrm{rank}(\\Phi) \\approx 0.2481$ for a two-state qubit-qutrit system
        """

        self.perp = Perp([self.Psi(i) for i in range(2)])

        res = rank(self.system, self.D, self.r, tries=2)

        self.assertIsInstance(
            res, float, msg="Entanglement rank result should be a float"
        )

        self.assertGreater(res, 0, msg="Entanglement rank should be positive")

        self.assertAlmostEqual(
            res, 0.2481, delta=1e-4, msg="Entanglement rank value mismatch"
        )


class QAOATests(Question):
    """
    QAOA solve: output structure, probabilities, optional func.
    """

    QUBO = {
        (0, 0): 1.0,
        (1, 1): 1.0,
        (0, 1): -2.0,
    }

    def test_solve_keys(self):
        """
        solve() returns 'solution' and 'probabilities' without func
        """
        from qudit.algo.qaoa import QAOA

        pt.manual_seed(0)

        q = QAOA(d=2, wires=2, qubo=self.QUBO, layers=1)

        result = q.solve(steps=20)

        self.assertIn("solution", result, msg="QAOA result missing 'solution' key")

        self.assertIn(
            "probabilities", result, msg="QAOA result missing 'probabilities' key"
        )

        self.assertNotIn(
            "value", result, msg="QAOA result should not have 'value' key without func"
        )

    def test_solve_solution_shape(self):
        """
        solution is a list of 2 bits
        """

        from qudit.algo.qaoa import QAOA

        pt.manual_seed(0)

        q = QAOA(d=2, wires=2, qubo=self.QUBO, layers=1)
        result = q.solve(steps=20)

        self.assertEqual(
            len(result["solution"]), 2, msg="QAOA solution should have 2 bits"
        )
        for bit in result["solution"]:
            self.assertIn(bit, [0, 1], msg="QAOA solution bits must be 0 or 1")

    def test_solve_probabilities_sum(self):
        """

        probabilities sum to 1
        """
        from qudit.algo.qaoa import QAOA

        pt.manual_seed(0)

        q = QAOA(d=2, wires=2, qubo=self.QUBO, layers=1)
        result = q.solve(steps=20)

        self.assertAlmostEqual(
            float(result["probabilities"].sum().item()),
            1.0,
            places=5,
            msg="QAOA probabilities should sum to 1",
        )

    def test_solve_with_func(self):
        """
        'value' key present when func is provided
        """
        from qudit.algo.qaoa import QAOA, Energy

        pt.manual_seed(0)

        def energy(qubo, x):
            return sum(
                qubo.get((i, j), 0.0) * x[i] * x[j] for i in range(2) for j in range(2)
            )

        q = QAOA(d=2, wires=2, qubo=self.QUBO, layers=1)
        result = q.solve(func=energy, steps=20)

        self.assertIn(
            "value",
            result,
            msg="QAOA result should have 'value' key when func provided",
        )


class ClockSolverTests(Question):
    """
    ClockSolver: qubit and qutrit output structure, probabilities.
    """

    QUBO = {
        (0, 0): 1.0,
        (1, 1): 1.0,
        (0, 1): -2.0,
    }

    def test_solve_qubit_keys(self):
        """
        ClockSolver(d=2) solve() returns 'solution' and 'probabilities'
        """
        from qudit.algo.qaoa import ClockSolver

        pt.manual_seed(0)

        cs = ClockSolver(d=2, wires=2, qubo=self.QUBO, layers=1)

        result = cs.solve(steps=20)

        self.assertIn(
            "solution", result, msg="ClockSolver result missing 'solution' key"
        )

        self.assertIn(
            "probabilities",
            result,
            msg="ClockSolver result missing 'probabilities' key",
        )

    def test_solve_qubit_solution_bits(self):
        """
        ClockSolver(d=2) solution contains valid bits
        """
        from qudit.algo.qaoa import ClockSolver

        pt.manual_seed(0)

        cs = ClockSolver(d=2, wires=2, qubo=self.QUBO, layers=1)
        result = cs.solve(steps=20)

        self.assertEqual(
            len(result["solution"]),
            2,
            msg="ClockSolver(d=2) solution should have 2 digits",
        )
        for digit in result["solution"]:
            self.assertIn(digit, [0, 1], msg="ClockSolver(d=2) digits must be 0 or 1")

    def test_solve_qutrit_solution_shape(self):
        """
        ClockSolver(d=3) solution digits are in {0,1,2}
        """
        from qudit.algo.qaoa import ClockSolver

        pt.manual_seed(0)
        ham = [(1.0, "Z", [0]), (1.0, "Z", [1])]

        cs = ClockSolver(d=3, wires=2, hamiltonian=ham, layers=1)
        result = cs.solve(steps=20)

        self.assertEqual(
            len(result["solution"]),
            2,
            msg="ClockSolver(d=3) solution should have 2 digits",
        )
        for digit in result["solution"]:
            self.assertIn(
                digit, [0, 1, 2], msg="ClockSolver(d=3) digits must be in {0,1,2}"
            )

    def test_solve_probabilities_sum(self):
        """
        ClockSolver probabilities sum to 1
        """
        from qudit.algo.qaoa import ClockSolver

        pt.manual_seed(0)

        cs = ClockSolver(d=2, wires=2, qubo=self.QUBO, layers=1)
        result = cs.solve(steps=20)

        self.assertAlmostEqual(
            float(result["probabilities"].sum().item()),
            1.0,
            places=5,
            msg="ClockSolver probabilities should sum to 1",
        )

    def test_solve_with_func(self):
        """
        'value' key present when func provided to ClockSolver
        """
        from qudit.algo.qaoa import ClockSolver

        def energy(qubo, x):
            return sum(
                qubo.get((i, j), 0.0) * x[i] * x[j] for i in range(2) for j in range(2)
            )

        pt.manual_seed(0)

        cs = ClockSolver(d=2, wires=2, qubo=self.QUBO, layers=1)
        result = cs.solve(func=energy, steps=20)

        self.assertIn(
            "value",
            result,
            msg="ClockSolver result should have 'value' key when func provided",
        )


if __name__ == "__main__":
    runner = Exam(
        name="Qudit Algo Tests",
        desc="Validation of stabilizer codes, entanglement rank, QAOA, and ClockSolver",
        file="algo.md",
    )
    runner.run(load(StabiliserCode))
    runner.run(load(EntanglementRank))
    runner.run(load(QAOATests))
    runner.run(load(ClockSolverTests))
