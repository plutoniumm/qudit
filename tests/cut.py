from MDR import Exam, load, Question
import sys

sys.path.append("..")

from qudit import Circuit
from qudit.cut import Coupler
import numpy as np
import torch


def _tvd(p, q):
    keys = set(p) | set(q)
    return sum(abs(p.get(k, 0) - q.get(k, 0)) for k in keys) / 2


def _full_probs(qc):
    from qudit.circuit.utils import to_mixed
    ket0 = torch.zeros(qc.width, dtype=torch.complex64)
    ket0[0] = 1.0
    out = qc(ket0).detach().reshape(-1).abs().pow(2)
    return {"".join(str(d) for d in to_mixed(i, qc.dims_)): float(out[i]) for i in range(qc.width)}


class CouplerDecomposition(Question):
    """
    Coupler reconstructs CX exactly from its decomposition terms for both methods.
    """

    def test_svd_reconstruct_2x2(self):
        """
        SVD decomposition of CX(2,2) reconstructs the gate to within 1e-5.
        """
        c = Coupler(2, 2, method="optimal", threshold=1e-8)
        err = (c.CX - c.reconstruct()).abs().max().item()
        self.assertAlmostEqual(err, 0.0, places=5, msg="SVD reconstruct failed for (2,2)")

    def test_svd_reconstruct_2x3(self):
        """
        SVD decomposition of CX(2,3) reconstructs the gate to within 1e-5.
        """
        c = Coupler(2, 3, method="optimal", threshold=1e-8)
        err = (c.CX - c.reconstruct()).abs().max().item()
        self.assertAlmostEqual(err, 0.0, places=5, msg="SVD reconstruct failed for (2,3)")

    def test_svd_reconstruct_3x3(self):
        """
        SVD decomposition of CX(3,3) reconstructs the gate to within 1e-5.
        """
        c = Coupler(3, 3, method="optimal", threshold=1e-8)
        err = (c.CX - c.reconstruct()).abs().max().item()
        self.assertAlmostEqual(err, 0.0, places=5, msg="SVD reconstruct failed for (3,3)")

    def test_basis_reconstruct_2x2(self):
        """
        Gell-Mann basis decomposition of CX(2,2) reconstructs the gate to within 1e-4.
        """
        c = Coupler(2, 2, method="basis", threshold=1e-8)
        err = (c.CX - c.reconstruct()).abs().max().item()
        self.assertAlmostEqual(err, 0.0, places=4, msg="Basis reconstruct failed for (2,2)")

    def test_basis_reconstruct_3x3(self):
        """
        Gell-Mann basis decomposition of CX(3,3) reconstructs the gate to within 1e-4.
        """
        c = Coupler(3, 3, method="basis", threshold=1e-8)
        err = (c.CX - c.reconstruct()).abs().max().item()
        self.assertAlmostEqual(err, 0.0, places=4, msg="Basis reconstruct failed for (3,3)")

    def test_terms_always_operators(self):
        """
        Both methods yield (Operator, Operator, float) tuples — uniform term format.
        """
        from qudit.circuit.gates import Operator
        for method in ("optimal", "basis"):
            c = Coupler(2, 3, method=method, threshold=1e-8)
            for opA, opB, coeff in c.terms:
                self.assertIsInstance(opA, Operator, msg=f"opA not Operator in {method}")
                self.assertIsInstance(opB, Operator, msg=f"opB not Operator in {method}")
                self.assertIsInstance(coeff, float, msg=f"coeff not float in {method}")


class VectorCut(Question):
    """
    run_cut() on a VECTOR mode circuit matches the full circuit probability distribution.
    """

    D1, D2 = 2, 3

    def _make(self, cut: bool):
        coupler = Coupler(self.D1, self.D2, method="optimal", threshold=1e-8)
        qc = Circuit(4, dim=[self.D1, self.D2, self.D1, self.D2])
        G1 = qc.gates[self.D1]
        qc.gate(G1.H, [0])
        qc.gate(G1.H, [2])
        qc.gate(coupler.CX, [0, 1])
        if cut:
            qc.cut(coupler, [2, 3])
        else:
            qc.gate(coupler.CX, [2, 3])
        return qc, coupler

    def test_tvd_optimal(self):
        """
        SVD cut TVD vs full circuit is below 1e-4.
        """
        qc_full, _ = self._make(cut=False)
        qc_cut, _ = self._make(cut=True)
        self.assertLess(_tvd(_full_probs(qc_full), qc_cut.run_cut()), 1e-4)

    def test_tvd_basis(self):
        """
        Gell-Mann basis cut TVD vs full circuit is below 1e-4.
        """
        coupler = Coupler(self.D1, self.D2, method="basis", threshold=1e-8)
        qc_full = Circuit(4, dim=[self.D1, self.D2, self.D1, self.D2])
        qc_cut = Circuit(4, dim=[self.D1, self.D2, self.D1, self.D2])
        for qc in (qc_full, qc_cut):
            G1 = qc.gates[self.D1]
            qc.gate(G1.H, [0])
            qc.gate(G1.H, [2])
            qc.gate(coupler.CX, [0, 1])
        qc_full.gate(coupler.CX, [2, 3])
        qc_cut.cut(coupler, [2, 3])
        self.assertLess(_tvd(_full_probs(qc_full), qc_cut.run_cut()), 1e-4)

    def test_probs_sum_to_one(self):
        """
        Reconstructed cut distribution sums to 1.
        """
        _, coupler = self._make(cut=False)
        qc_cut, _ = self._make(cut=True)
        total = sum(qc_cut.run_cut().values())
        self.assertAlmostEqual(total, 1.0, places=4)

    def test_uniform_qubit(self):
        """
        d=2 qubit circuit cut matches full distribution.
        """
        coupler = Coupler(2, 2, method="optimal", threshold=1e-8)
        qc_full = Circuit(4, dim=2)
        qc_cut = Circuit(4, dim=2)
        for qc in (qc_full, qc_cut):
            G = qc.gates[2]
            qc.gate(G.H, [0])
            qc.gate(coupler.CX, [0, 1])
        qc_full.gate(coupler.CX, [2, 3])
        qc_cut.cut(coupler, [2, 3])
        self.assertLess(_tvd(_full_probs(qc_full), qc_cut.run_cut()), 1e-4)

    def test_mixed_dims_2x3(self):
        """
        Mixed-dimension cut [D1=2, D2=3] matches the full circuit.
        """
        qc_full, _ = self._make(cut=False)
        qc_cut, _ = self._make(cut=True)
        self.assertLess(_tvd(_full_probs(qc_full), qc_cut.run_cut()), 1e-4)


class MatrixCut(Question):
    """
    run_cut() on a MATRIX (density matrix) mode circuit matches the full circuit diagonal.
    """

    D1, D2 = 2, 3

    def test_tvd_matrix_mode(self):
        """
        MATRIX mode cut TVD vs full circuit density matrix diagonal is below 1e-4.
        """
        from qudit.circuit.index import Mode
        from qudit.circuit.utils import to_mixed

        coupler = Coupler(self.D1, self.D2, method="optimal", threshold=1e-8)

        def build(mode, cut):
            qc = Circuit(4, dim=[self.D1, self.D2, self.D1, self.D2], mode=mode)
            G1 = qc.gates[self.D1]
            qc.gate(G1.H, [0])
            qc.gate(G1.H, [2])
            qc.gate(coupler.CX, [0, 1])
            if cut:
                qc.cut(coupler, [2, 3])
            else:
                qc.gate(coupler.CX, [2, 3])
            return qc

        qc_full = build(Mode.MATRIX, cut=False)
        ket0 = torch.zeros(qc_full.width, dtype=torch.complex64)
        ket0[0] = 1.0
        rho = qc_full(ket0).detach().numpy()
        p_full = {
            "".join(str(d) for d in to_mixed(i, qc_full.dims_)): float(np.real(rho[i, i]))
            for i in range(qc_full.width)
        }

        qc_cut = build(Mode.MATRIX, cut=True)
        p_cut = qc_cut.run_cut()

        self.assertLess(_tvd(p_full, p_cut), 1e-4)


class CutValidation(Question):
    """
    run_cut() raises correctly on invalid configurations.
    """

    def test_no_cuts_raises(self):
        """
        run_cut() on a circuit with no cuts defined raises ValueError.
        """
        qc = Circuit(2, dim=2)
        qc.gate(qc.gates[2].H, [0])
        with self.assertRaises(ValueError):
            qc.run_cut()

    def test_non_adjacent_cut_raises(self):
        """
        A cut on non-adjacent wires (gap > 1) raises ValueError.
        """
        coupler = Coupler(2, 2)
        qc = Circuit(4, dim=2)
        qc.cut(coupler, [0, 2])
        with self.assertRaises(ValueError):
            qc.run_cut()

    def test_spanning_gate_raises(self):
        """
        A non-cut gate whose wires span both partitions raises ValueError.
        """
        coupler = Coupler(2, 2)
        qc = Circuit(4, dim=2)
        qc.gate(qc.gates[2].CX, [1, 2])
        qc.cut(coupler, [2, 3])
        with self.assertRaises(ValueError):
            qc.run_cut()


class LittleEndianTests(Question):
    """
    LittleEndian reverses qudit wire ordering on statevectors (uniform and mixed base).
    """

    def test_uniform_base_reversal(self):
        """
        |10⟩ (index 2 in d=2, 2-wire big-endian) maps to index 1 after flip.
        """
        from qudit.circuit.utils import LittleEndian
        state = torch.zeros(4)
        state[2] = 1.0
        out = LittleEndian(state, base=2)
        self.assertEqual(out[1].item(), 1.0)

    def test_mixed_base_reversal(self):
        """
        Mixed base [2,3]: |1,0⟩ (index 3) maps to index 1 after flip.
        """
        from qudit.circuit.utils import LittleEndian
        state = torch.zeros(6)
        state[3] = 1.0
        out = LittleEndian(state, base=[2, 3])
        self.assertEqual(out[1].item(), 1.0)

    def test_involution(self):
        """
        Applying LittleEndian twice returns the original vector.
        """
        from qudit.circuit.utils import LittleEndian
        state = torch.rand(6)
        state = state / state.norm()
        roundtrip = LittleEndian(LittleEndian(state, [2, 3]), [2, 3])
        self.assertTrue(torch.allclose(state, roundtrip, atol=1e-6))

    def test_d3_reversal(self):
        """
        d=3, 2-wire: |2,0⟩ (index 6) maps to |0,2⟩ (index 2) after flip.
        """
        from qudit.circuit.utils import LittleEndian
        state = torch.zeros(9)
        state[6] = 1.0   # |2,0⟩ = 2*3+0 = 6
        out = LittleEndian(state, base=3)
        self.assertEqual(out[2].item(), 1.0)   # |0,2⟩ = 0*3+2 = 2


if __name__ == "__main__":
    runner = Exam(
        name="Circuit Cut Tests",
        desc="Coupler decomposition and run_cut() correctness",
        file="cut.md",
    )
    runner.run(load(CouplerDecomposition))
    runner.run(load(VectorCut))
    runner.run(load(MatrixCut))
    runner.run(load(CutValidation))
    runner.run(load(LittleEndianTests))
