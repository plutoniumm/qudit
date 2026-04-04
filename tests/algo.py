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
        self.assertEqual(s.num_states, 2)

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
        self.assertIsInstance(res, float)
        self.assertGreater(res, 0)
        self.assertAlmostEqual(res, 0.2481, delta=1e-4)


if __name__ == "__main__":
    runner = Exam(
        name="Qudit Algo Tests",
        desc="Validation of stabilizer codes and entanglement rank",
        file="algo.md",
    )
    runner.run(load(StabiliserCode))
    runner.run(load(EntanglementRank))
