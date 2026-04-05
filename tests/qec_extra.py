from MDR import Exam, load, Question
import sys
import torch as pt

sys.path.append("..")

from qudit.qec.lib import Dutta3, Perfect
from qudit.qec.codes import Code


class CodeOrthonormality(Question):
    """
    Dutta3, Perfect codewords are orthonormal; fromStabilizers produces valid codes.
    """

    def test_dutta3_orthonormal(self):
        """
        Dutta3 codewords are orthonormal:
        ⟨0_L|0_L⟩ = ⟨1_L|1_L⟩ = 1, ⟨0_L|1_L⟩ = 0
        """
        s0, s1 = Dutta3().toTensor()

        self.assertAlmostEqual(pt.dot(s0, s0).real.item(), 1.0, places=6, msg="Dutta3 |0_L> not normalized")

        self.assertAlmostEqual(pt.dot(s1, s1).real.item(), 1.0, places=6, msg="Dutta3 |1_L> not normalized")

        self.assertAlmostEqual(pt.abs(pt.dot(s0, s1)).item(), 0.0, places=6, msg="Dutta3 codewords not orthogonal")

    def test_perfect_orthonormal(self):
        """
        [[5,1,3]] Perfect code codewords are orthonormal
        """
        s0, s1 = Perfect().toTensor()

        self.assertAlmostEqual(pt.dot(s0, s0).real.item(), 1.0, places=5, msg="Perfect |0_L> not normalized")

        self.assertAlmostEqual(pt.dot(s1, s1).real.item(), 1.0, places=5, msg="Perfect |1_L> not normalized")

        self.assertAlmostEqual(pt.abs(pt.dot(s0, s1)).item(), 0.0, places=5, msg="Perfect codewords not orthogonal")

    def test_perfect_norm_each(self):
        """
        Each Perfect codeword has unit norm
        """
        code = Perfect().toTensor()
        for cw in code:
            self.assertAlmostEqual(pt.linalg.norm(cw.float()).item(), 1.0, places=5, msg="Each Perfect codeword should have unit norm")

    def test_dutta3_span_3qubit(self):
        """
        Dutta3 lives in 2^3=8-dimensional Hilbert space
        """
        self.assertEqual(Dutta3().toTensor().shape[1], 8, msg="Dutta3 should live in 2^3=8-dimensional space")

    def test_perfect_span_5qubit(self):
        """
        Perfect code lives in 2^5=32-dimensional Hilbert space
        """
        self.assertEqual(Perfect().toTensor().shape[1], 32, msg="Perfect code should live in 2^5=32-dimensional space")

    def test_from_stabilizers_3qubit_bitflip(self):
        """
        3-qubit bit-flip code stabilizers [ZZI, IZZ] give valid orthonormal codewords.
        Logical codewords are |000⟩ and |111⟩.
        """
        code = Code.fromStabilizers(["ZZI", "IZZ"], method="svd")
        cws = code.toTensor()

        self.assertEqual(cws.shape[0], 2, msg="3-qubit bit-flip code should have 2 codewords")

        self.assertEqual(cws.shape[1], 8, msg="3-qubit code codewords should be in 2^3=8-dim space")
        norms = pt.linalg.norm(cws.float(), dim=1)
        for n in norms:
            self.assertAlmostEqual(n.item(), 1.0, places=5, msg="3-qubit bit-flip codewords should be normalized")
        inner = pt.abs(cws[0] @ cws[1].conj()).item()

        self.assertAlmostEqual(inner, 0.0, places=5, msg="3-qubit bit-flip codewords should be orthogonal")

    def test_from_stabilizers_codewords_are_000_111(self):
        """
        3-qubit bit-flip code codewords should be |000⟩ and |111⟩
        """
        import torch as pt

        code = Code.fromStabilizers(["ZZI", "IZZ"], method="svd")
        cws = code.toTensor()
        for cw in cws:
            nonzero = int((pt.abs(cw) > 1e-6).sum().item())

            self.assertEqual(nonzero, 1, msg="3-qubit bit-flip codeword should be a computational basis state")
        # One should be at index 0 (|000⟩) and one at index 7 (|111⟩)
        indices = {
            int(pt.argmax(pt.abs(cws[0])).item()),
            int(pt.argmax(pt.abs(cws[1])).item()),
        }

        self.assertEqual(indices, {0, 7}, msg="Codewords should be |000> (index 0) and |111> (index 7)")


if __name__ == "__main__":
    runner = Exam(
        name="QEC Code Tests",
        desc="Orthonormality of Dutta3, Perfect codewords and fromStabilizers",
        file="qec_extra.md",
    )
    runner.run(load(CodeOrthonormality))
