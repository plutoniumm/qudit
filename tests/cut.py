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

    return {
        "".join(str(d) for d in to_mixed(i, qc.dims_)): float(out[i])
        for i in range(qc.width)
    }


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

        self.assertAlmostEqual(
            err, 0.0, places=5, msg="SVD reconstruct failed for (2,2)"
        )

    def test_svd_reconstruct_2x3(self):
        """
        SVD decomposition of CX(2,3) reconstructs the gate to within 1e-5.
        """

        c = Coupler(2, 3, method="optimal", threshold=1e-8)
        err = (c.CX - c.reconstruct()).abs().max().item()

        self.assertAlmostEqual(
            err, 0.0, places=5, msg="SVD reconstruct failed for (2,3)"
        )

    def test_svd_reconstruct_3x3(self):
        """
        SVD decomposition of CX(3,3) reconstructs the gate to within 1e-5.
        """

        c = Coupler(3, 3, method="optimal", threshold=1e-8)
        err = (c.CX - c.reconstruct()).abs().max().item()

        self.assertAlmostEqual(
            err, 0.0, places=5, msg="SVD reconstruct failed for (3,3)"
        )

    def test_basis_reconstruct_2x2(self):
        """
        Gell-Mann basis decomposition of CX(2,2) reconstructs the gate to within 1e-4.
        """

        c = Coupler(2, 2, method="basis", threshold=1e-8)
        err = (c.CX - c.reconstruct()).abs().max().item()

        self.assertAlmostEqual(
            err, 0.0, places=4, msg="Basis reconstruct failed for (2,2)"
        )

    def test_basis_reconstruct_3x3(self):
        """
        Gell-Mann basis decomposition of CX(3,3) reconstructs the gate to within 1e-4.
        """

        c = Coupler(3, 3, method="basis", threshold=1e-8)
        err = (c.CX - c.reconstruct()).abs().max().item()

        self.assertAlmostEqual(
            err, 0.0, places=4, msg="Basis reconstruct failed for (3,3)"
        )

    def test_terms_always_operators(self):
        """
        Both methods yield (Operator, Operator, numeric) tuples — uniform structure.
        SVD coefficients are float; basis coefficients may be complex for non-Hermitian ops.
        """
        from qudit.circuit.gates import Operator

        for method in ("optimal", "basis"):
            c = Coupler(2, 3, method=method, threshold=1e-8)
            for opA, opB, coeff in c.terms:
                self.assertIsInstance(

                    opA, Operator, msg=f"opA not Operator in {method}"
                )

                self.assertIsInstance(

                    opB, Operator, msg=f"opB not Operator in {method}"
                )

                self.assertIsInstance(
                    coeff, (int, float, complex), msg=f"coeff not numeric in {method}"
                )


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

        self.assertLess(_tvd(_full_probs(qc_full), qc_cut.run_cut()), 1e-4, msg="SVD cut TVD too high")

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

        self.assertLess(_tvd(_full_probs(qc_full), qc_cut.run_cut()), 1e-4, msg="Basis cut TVD too high")

    def test_probs_sum_to_one(self):
        """
        Reconstructed cut distribution sums to 1.
        """

        _, coupler = self._make(cut=False)
        qc_cut, _ = self._make(cut=True)
        total = sum(qc_cut.run_cut().values())

        self.assertAlmostEqual(total, 1.0, places=4, msg="Cut probs don't sum to 1")

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

        self.assertLess(_tvd(_full_probs(qc_full), qc_cut.run_cut()), 1e-4, msg="Qubit cut TVD too high")

    def test_mixed_dims_2x3(self):
        """
        Mixed-dimension cut [D1=2, D2=3] matches the full circuit.

        """
        qc_full, _ = self._make(cut=False)
        qc_cut, _ = self._make(cut=True)

        self.assertLess(_tvd(_full_probs(qc_full), qc_cut.run_cut()), 1e-4, msg="Mixed-dim cut TVD too high")


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
        rho = qc_full(ket0).detach().resolve_conj().numpy()
        p_full = {
            "".join(str(d) for d in to_mixed(i, qc_full.dims_)): float(
                np.real(rho[i, i])
            )
            for i in range(qc_full.width)
        }

        qc_cut = build(Mode.MATRIX, cut=True)
        p_cut = qc_cut.run_cut()

        self.assertLess(_tvd(p_full, p_cut), 1e-4, msg="Matrix mode cut TVD too high")


class NonAdjacentCut(Question):
    """
    Non-adjacent cut [c_A, c_B] with explicit split= assigning gap wires to a partition.
    """

    def test_gap_in_A(self):
        """
        Cut [1,3] split=2: partA={0,1}, partB={2,3,4}. Gate on wire 2 (in B) is legal.
        Full circuit matches cut distribution to within 1e-4 TVD.
        """
        coupler = Coupler(2, 2)
        qc_full = Circuit(5, dim=2)
        qc_cut = Circuit(5, dim=2)
        for qc in (qc_full, qc_cut):
            G = qc.gates[2]
            qc.gate(G.H, [0])
            qc.gate(G.H, [3])
            qc.gate(coupler.CX, [0, 1])
        qc_full.gate(coupler.CX, [1, 3])
        qc_cut.cut(coupler, [1, 3], split=2)

        self.assertLess(
            _tvd(_full_probs(qc_full), qc_cut.run_cut()),
            1e-4,
            msg="Non-adjacent cut (split=2) TVD too high",
        )

    def test_gap_in_B(self):
        """
        Cut [1,3] split=3: partA={0,1,2}, partB={3,4}. Gap wire 2 is in A.
        Full circuit matches cut distribution to within 1e-4 TVD.
        """
        coupler = Coupler(2, 2)
        qc_full = Circuit(5, dim=2)
        qc_cut = Circuit(5, dim=2)
        for qc in (qc_full, qc_cut):
            G = qc.gates[2]
            qc.gate(G.H, [0])
            qc.gate(G.H, [3])
            qc.gate(coupler.CX, [0, 1])
            qc.gate(G.H, [2])
        qc_full.gate(coupler.CX, [1, 3])
        qc_cut.cut(coupler, [1, 3], split=3)

        self.assertLess(
            _tvd(_full_probs(qc_full), qc_cut.run_cut()),
            1e-4,
            msg="Non-adjacent cut (split=3) TVD too high",
        )

    def test_missing_split_raises(self):
        """
        Non-adjacent cut without split= raises ValueError.
        """
        coupler = Coupler(2, 2)
        qc = Circuit(5, dim=2)
        with self.assertRaises(ValueError):
            qc.cut(coupler, [1, 3])

    def test_trajectory_raises(self):
        """
        run_cut() on a TRAJECTORY mode circuit raises NotImplementedError.
        """
        from qudit.circuit.index import Mode
        from qudit.noise.model import PhysicalNoise

        coupler = Coupler(2, 2)
        noise = PhysicalNoise(T1=1000, T2=500)
        qc = Circuit(4, dim=2, mode=Mode.TRAJECTORY, noise=noise)
        qc.cut(coupler, [1, 2])
        with self.assertRaises(NotImplementedError):
            qc.run_cut()


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

    def test_inverted_cut_raises(self):
        """
        A cut where c_B <= c_A raises ValueError at cut() time.
        """
        coupler = Coupler(2, 2)
        qc = Circuit(4, dim=2)
        with self.assertRaises(ValueError):
            qc.cut(coupler, [2, 1])

    def test_spanning_gate_raises(self):
        """
        A non-cut gate with one wire in each partition raises ValueError.
        cut at [2,3] → partA={0,1,2}, partB={3}; gate at [1,3] spans both.
        """
        coupler = Coupler(2, 2)
        qc = Circuit(4, dim=2)
        qc.gate(qc.gates[2].CX, [1, 3])
        qc.cut(coupler, [2, 3])
        with self.assertRaises(ValueError):
            qc.run_cut()

    def test_post_cut_gate_A(self):
        """
        A gate on partition A after cut() is correctly applied after the cut op.
        H on wire 0 before and after the cut — full circuit matches cut distribution.
        """
        coupler = Coupler(2, 2)
        qc_full = Circuit(4, dim=2)
        qc_cut = Circuit(4, dim=2)
        G = qc_full.gates[2]
        for qc in (qc_full, qc_cut):
            qc.gate(G.H, [0])
            qc.gate(coupler.CX, [0, 1])
        qc_full.gate(coupler.CX, [2, 3])
        qc_full.gate(G.H, [0])
        qc_cut.cut(coupler, [2, 3])
        qc_cut.gate(G.H, [0])

        self.assertLess(
            _tvd(_full_probs(qc_full), qc_cut.run_cut()),
            1e-4,
            msg="Post-cut gate on A TVD too high",
        )

    def test_post_cut_gate_B(self):
        """
        A gate on partition B after cut() is correctly applied after the cut op.
        H on wire 3 after the cut — full circuit matches cut distribution.
        """
        coupler = Coupler(2, 2)
        qc_full = Circuit(4, dim=2)
        qc_cut = Circuit(4, dim=2)
        G = qc_full.gates[2]
        for qc in (qc_full, qc_cut):
            qc.gate(G.H, [0])
            qc.gate(coupler.CX, [0, 1])
        qc_full.gate(coupler.CX, [2, 3])
        qc_full.gate(G.H, [3])
        qc_cut.cut(coupler, [2, 3])
        qc_cut.gate(G.H, [3])

        self.assertLess(
            _tvd(_full_probs(qc_full), qc_cut.run_cut()),
            1e-4,
            msg="Post-cut gate on B TVD too high",
        )

    def test_post_cut_gates_both(self):
        """
        Gates on both partitions after cut() — full circuit matches cut distribution.
        """
        coupler = Coupler(2, 2)
        qc_full = Circuit(4, dim=2)
        qc_cut = Circuit(4, dim=2)
        G = qc_full.gates[2]
        for qc in (qc_full, qc_cut):
            qc.gate(G.H, [0])
            qc.gate(coupler.CX, [0, 1])
        qc_full.gate(coupler.CX, [2, 3])
        qc_full.gate(G.H, [0])
        qc_full.gate(G.H, [3])
        qc_cut.cut(coupler, [2, 3])
        qc_cut.gate(G.H, [0])
        qc_cut.gate(G.H, [3])

        self.assertLess(
            _tvd(_full_probs(qc_full), qc_cut.run_cut()),
            1e-4,
            msg="Post-cut gates on both partitions TVD too high",
        )


class MultiCut(Question):
    """
    run_cut() with multiple cuts reconstructs the joint distribution via tensor contraction.
    """

    def test_two_cuts_basic(self):
        """
        Two adjacent cuts on a 6-wire circuit (partitions AAA|BB|C) match the full circuit.
        """
        c1 = Coupler(2, 2)
        c2 = Coupler(2, 2)
        qc_full = Circuit(6, dim=2)
        qc_cut = Circuit(6, dim=2)
        G = qc_full.gates[2]
        for qc in (qc_full, qc_cut):
            qc.gate(G.H, [0])
            qc.gate(G.H, [3])
            qc.gate(G.H, [5])
            qc.gate(c1.CX, [0, 1]) if qc is qc_full else qc.gate(c1.CX, [0, 1])
        qc_full.gate(c1.CX, [2, 3])
        qc_full.gate(c2.CX, [4, 5])
        qc_cut.cut(c1, [2, 3])
        qc_cut.cut(c2, [4, 5])

        self.assertLess(
            _tvd(_full_probs(qc_full), qc_cut.run_cut()),
            1e-4,
            msg="Two-cut TVD too high",
        )

    def test_two_cuts_with_middle_gates(self):
        """
        Gates in the middle partition (between two cuts) are correctly segmented.
        """
        c1 = Coupler(2, 2)
        c2 = Coupler(2, 2)
        qc_full = Circuit(6, dim=2)
        qc_cut = Circuit(6, dim=2)
        G = qc_full.gates[2]
        for qc in (qc_full, qc_cut):
            qc.gate(G.H, [0])
            qc.gate(G.H, [3])
        qc_full.gate(c1.CX, [2, 3])
        qc_full.gate(G.H, [3])
        qc_full.gate(G.H, [3])
        qc_full.gate(c2.CX, [4, 5])
        qc_cut.cut(c1, [2, 3])
        qc_cut.gate(G.H, [3])
        qc_cut.gate(G.H, [3])
        qc_cut.cut(c2, [4, 5])

        self.assertLess(
            _tvd(_full_probs(qc_full), qc_cut.run_cut()),
            1e-4,
            msg="Two-cut with middle partition gates TVD too high",
        )

    def test_two_cuts_post_cut_gates(self):
        """
        Post-cut gates on all three partitions are correctly placed.
        """
        c1 = Coupler(2, 2)
        c2 = Coupler(2, 2)
        qc_full = Circuit(6, dim=2)
        qc_cut = Circuit(6, dim=2)
        G = qc_full.gates[2]
        for qc in (qc_full, qc_cut):
            qc.gate(G.H, [0])
            qc.gate(G.H, [3])
            qc.gate(G.H, [5])
        qc_full.gate(c1.CX, [2, 3])
        qc_full.gate(c2.CX, [4, 5])
        qc_full.gate(G.H, [0])
        qc_full.gate(G.H, [3])
        qc_full.gate(G.H, [5])
        qc_cut.cut(c1, [2, 3])
        qc_cut.cut(c2, [4, 5])
        qc_cut.gate(G.H, [0])
        qc_cut.gate(G.H, [3])
        qc_cut.gate(G.H, [5])

        self.assertLess(
            _tvd(_full_probs(qc_full), qc_cut.run_cut()),
            1e-4,
            msg="Two-cut post-cut gates TVD too high",
        )

    def test_two_cuts_probs_sum_to_one(self):
        """
        Reconstructed multi-cut distribution sums to 1.
        """
        c1 = Coupler(2, 2)
        c2 = Coupler(2, 2)
        qc = Circuit(6, dim=2)
        G = qc.gates[2]
        qc.gate(G.H, [0])
        qc.gate(G.H, [3])
        qc.cut(c1, [2, 3])
        qc.cut(c2, [4, 5])
        total = sum(qc.run_cut().values())

        self.assertAlmostEqual(total, 1.0, places=4, msg="Multi-cut probs don't sum to 1")

    def test_out_of_order_cuts_raises(self):
        """
        Cuts added in reverse wire order raise ValueError.
        """
        c1 = Coupler(2, 2)
        c2 = Coupler(2, 2)
        qc = Circuit(6, dim=2)
        qc.cut(c2, [4, 5])
        qc.cut(c1, [2, 3])
        with self.assertRaises(ValueError, msg="Out-of-order cuts should raise"):
            qc.run_cut()

    def test_qiskit_aabb_shape(self):
        """
        Qiskit Tutorial 01 shape: 4q EfficientSU2 (1 rep), AABB, two gate cuts.
        rot_layer → CX(0,1) → [CUT CX(1,2)] → CX(2,3) → rot_layer: single cut handled.
        Here we cut CX(1,2) AND simulate a second boundary via a no-op second cut.
        Full two-cut: A={0,1}, B={2}, C={3}, cuts at CX(1,2) and CX(2,3).
        """
        import math

        def RY(t):
            c, s = math.cos(t / 2), math.sin(t / 2)

            return torch.tensor([[c, -s], [s, c]], dtype=torch.complex64)

        theta = 0.4
        c1 = Coupler(2, 2)
        c2 = Coupler(2, 2)

        def make(cut):
            qc = Circuit(4, dim=2)
            G = qc.gates[2]
            for w in range(4):
                qc.gate(RY(theta), [w])
            qc.gate(G.CX, [0, 1])
            if cut:
                qc.cut(c1, [1, 2])
                qc.gate(RY(theta), [2])
                qc.cut(c2, [2, 3])
            else:
                qc.gate(c1.CX, [1, 2])
                qc.gate(RY(theta), [2])
                qc.gate(c2.CX, [2, 3])
            for w in range(4):
                qc.gate(RY(theta), [w])

            return qc

        self.assertLess(
            _tvd(_full_probs(make(False)), make(True).run_cut()),
            1e-4,
            msg="Qiskit AABB two-cut TVD too high",
        )


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

        self.assertEqual(out[1].item(), 1.0, msg="Uniform base flip failed for |10⟩")

    def test_mixed_base_reversal(self):
        """
        Mixed base [2,3]: |1,0⟩ (index 3) maps to index 1 after flip.
        """
        from qudit.circuit.utils import LittleEndian


        state = torch.zeros(6)
        state[3] = 1.0
        out = LittleEndian(state, base=[2, 3])

        self.assertEqual(out[1].item(), 1.0, msg="Mixed base flip failed for |1,0⟩")

    def test_involution_uniform(self):
        """
        For uniform base, LittleEndian is self-inverse: flip(flip(v)) = v.
        Mixed bases are NOT self-inverse (requires flip with reversed bases).
        """
        from qudit.circuit.utils import LittleEndian


        state = torch.rand(8)
        state = state / state.norm()
        roundtrip = LittleEndian(LittleEndian(state, base=2), base=2)

        self.assertTrue(torch.allclose(state, roundtrip, atol=1e-6), msg="LittleEndian involution failed for uniform base")

    def test_d3_reversal(self):
        """
        d=3, 2-wire: |2,0⟩ (index 6) maps to |0,2⟩ (index 2) after flip.
        """
        from qudit.circuit.utils import LittleEndian


        state = torch.zeros(9)
        state[6] = 1.0  # |2,0⟩ = 2*3+0 = 6
        out = LittleEndian(state, base=3)

        self.assertEqual(out[2].item(), 1.0, msg="d=3 flip failed for |2,0⟩")  # |0,2⟩ = 0*3+2 = 2


if __name__ == "__main__":
    runner = Exam(
        name="Circuit Cut Tests",
        desc="Coupler decomposition and run_cut() correctness",
        file="cut.md",
    )
    runner.run(load(CouplerDecomposition))
    runner.run(load(VectorCut))
    runner.run(load(MatrixCut))
    runner.run(load(NonAdjacentCut))
    runner.run(load(CutValidation))
    runner.run(load(MultiCut))
    runner.run(load(LittleEndianTests))
