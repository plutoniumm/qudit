from MDR import Exam, load, Question
import sys
import numpy as np
import torch as pt

sys.path.append("..")

from qudit.tools.tests import Space
from qudit.utils import Braket, Tensor, partial
from qudit.index import Basis


def make_bell():
    b = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)

    return np.outer(b, b.conj())


class SpaceTests(Question):
    """
    gramSchmidt, schmidtDecompose, schmidtRank, PPT.
    """

    def test_gramschmidt_orthonormal(self):
        """
        Gram-Schmidt output is orthonormal
        """
        vecs = pt.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        Q = Space.gramSchmidt(vecs)
        G = Q @ Q.conj().T

        for i in range(len(Q)):
            for j in range(len(Q)):

                expected = 1.0 if i == j else 0.0

                self.assertAlmostEqual(float(G[i, j].abs()), expected, places=5, msg=f"Gram matrix G[{i},{j}] should be {expected}")

    def test_schmidt_rank_product(self):

        """
        Product state has Schmidt rank 1
        """
        mat = pt.tensor([[1.0, 0.0], [0.0, 0.0]])

        self.assertEqual(Space.schmidtRank(mat), 1, msg="Product state should have Schmidt rank 1")


    def test_schmidt_rank_entangled(self):
        """
        Bell state coefficient matrix has Schmidt rank 2
        """
        mat = pt.tensor([[1.0, 0.0], [0.0, 1.0]]) / np.sqrt(2)

        self.assertEqual(Space.schmidtRank(mat), 2, msg="Bell state should have Schmidt rank 2")


    def test_schmidt_decompose_values(self):
        """
        schmidtDecompose returns correct singular values for Bell state
        """
        mat = pt.tensor([[1.0, 0.0], [0.0, 1.0]]) / np.sqrt(2)
        decomp = Space.schmidtDecompose(mat)

        self.assertAlmostEqual(float(decomp[0][0]), 1 / np.sqrt(2), places=5, msg="Bell state first Schmidt value should be 1/√2")

        self.assertAlmostEqual(float(decomp[1][0]), 1 / np.sqrt(2), places=5, msg="Bell state second Schmidt value should be 1/√2")

    def test_ppt_separable(self):
        """
        Product state |00⟩⟨00| satisfies PPT
        """
        # Use a 3x3 block-diagonal product state (sub=3 hardcoded in PPT)
        rho = pt.zeros((9, 9), dtype=pt.complex128)

        rho[0, 0] = 1.0

        self.assertTrue(Space.PPT(rho, 3), msg="Product state |00><00| should satisfy PPT")



class PartialTraceTests(Question):
    """
    partial.trace and partial.transpose with known values.
    """

    def test_trace_product_keep_A(self):
        """
        Tr_B(|00⟩⟨00|) = |0⟩⟨0|
        """

        rho = np.zeros((4, 4), dtype=complex)
        rho[0, 0] = 1.0

        rho_A = np.array(partial.trace(rho, 2, 2, keep="A"))

        self.assertAlmostEqual(float(np.abs(rho_A[0, 0])), 1.0, places=6, msg="Tr_B(|00><00|)[0,0] should be 1")

        self.assertAlmostEqual(float(np.abs(rho_A[1, 1])), 0.0, places=6, msg="Tr_B(|00><00|)[1,1] should be 0")

    def test_trace_bell_maximally_mixed(self):
        """
        Tr_B(Bell) = I/2
        """

        rho_B = np.array(partial.trace(make_bell(), 2, 2, keep="B"))


        self.assertAlmostEqual(float(np.abs(rho_B[0, 0])), 0.5, places=5, msg="Tr_A(Bell)[0,0] should be 0.5")

        self.assertAlmostEqual(float(np.abs(rho_B[1, 1])), 0.5, places=5, msg="Tr_A(Bell)[1,1] should be 0.5")

        self.assertAlmostEqual(float(np.abs(rho_B[0, 1])), 0.0, places=5, msg="Tr_A(Bell)[0,1] off-diagonal should be 0")

    def test_trace_preserves_total(self):
        """
        Partial trace of a density matrix has trace 1
        """

        rho_A = np.array(partial.trace(make_bell(), 2, 2, keep="A"))

        self.assertAlmostEqual(float(np.trace(rho_A).real), 1.0, places=5, msg="Partial trace of density matrix should have trace 1")

    def test_transpose_hermitian(self):
        """
        partial.transpose applied twice returns original
        """
        rho = make_bell()

        rho_pt = np.array(partial.transpose(rho, 2, 2))

        rho_pt2 = np.array(partial.transpose(rho_pt, 2, 2))

        self.assertTrue(np.allclose(rho, rho_pt2, atol=1e-6), msg="Applying partial transpose twice should return original")


class BraketTensorTests(Question):
    """
    Braket and Tensor utility functions.
    """


    def test_braket_self(self):
        """
        ⟨0|0⟩ = 1
        """
        B = Basis(2)
        ket0 = B(0)


        result = Braket(ket0, ket0)

        self.assertAlmostEqual(float(np.abs(result)), 1.0, places=6, msg="<0|0> should be 1")

    def test_braket_orthogonal(self):
        """
        ⟨0|1⟩ = 0
        """
        B = Basis(2)

        result = Braket(B(0), B(1))

        self.assertAlmostEqual(float(np.abs(result)), 0.0, places=6, msg="<0|1> should be 0")

    def test_tensor_product(self):
        """
        |0⟩ ⊗ |1⟩ has amplitude 1 at index 1
        """

        B = Basis(2)
        ket01 = Tensor(B(0), B(1))

        arr = np.array(ket01)

        self.assertAlmostEqual(float(np.abs(arr[1])), 1.0, places=6, msg="|0>⊗|1> should have amplitude 1 at index 1")

        self.assertAlmostEqual(float(np.abs(arr[0])), 0.0, places=6, msg="|0>⊗|1> should have amplitude 0 at index 0")

    def test_tensor_norm(self):
        """
        Tensor product of normalized states is normalized
        """
        B = Basis(2)

        result = Tensor(B(0), B(1))

        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=6, msg="Tensor product of normalized states should be normalized")


class SpaceExtendedTests(Question):
    """
    Extended Space tests: gramSchmidt with linearly dependent inputs,
    schmidtDecompose reconstruction, PPT on mixed state.
    """

    def test_gramschmidt_drops_dependent(self):
        """
        Gram-Schmidt drops linearly dependent vectors: 3 input vectors with 2 independent gives 2 output

        """
        # v3 = v1 + v2, so rank is 2
        vecs = pt.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])

        Q = Space.gramSchmidt(vecs)

        self.assertEqual(
            Q.shape[0], 2, msg="GS should return 2 orthonormal vectors for rank-2 input"
        )

    def test_schmidt_decompose_product_state(self):
        """

        Product state $|00\\rangle = |0\\rangle \\otimes |0\\rangle$ has Schmidt rank 1:
        only one non-zero singular value = 1.0
        """
        # |00> as 2x2 amplitude matrix: M[i,j] = <i|<j|psi> = delta_{i0}*delta_{j0}
        mat = pt.tensor([[1.0, 0.0], [0.0, 0.0]])

        decomp = Space.schmidtDecompose(mat)

        self.assertAlmostEqual(float(decomp[0][0]), 1.0, places=5, msg="Product state should have only one Schmidt value = 1.0")

        self.assertAlmostEqual(float(decomp[1][0]), 0.0, places=5, msg="Product state second Schmidt value should be 0")

    def test_schmidt_rank_ghz_in_matrix_form(self):
        """
        $GHZ(2,3)$ reshaped as a $3 \\times 3$ amplitude matrix has Schmidt rank 3
        (all computational basis states $|00\\rangle,|11\\rangle,|22\\rangle$ contribute)
        """
        from qudit.tools.states import GHZ

        ghz = GHZ(2, 3)


        mat = pt.tensor(np.array(ghz)).reshape(3, 3)

        self.assertEqual(
            Space.schmidtRank(mat), 3, msg="GHZ(2,3) Schmidt rank should be 3"
        )

    def test_ppt_maximally_mixed(self):
        """

        Maximally mixed state $I/9$ on qutrit⊗qutrit space satisfies PPT
        (separable state)
        """

        rho = pt.eye(9, dtype=pt.complex64) / 9.0

        self.assertTrue(
            Space.PPT(rho, 3), msg="Maximally mixed state should satisfy PPT"
        )


class PartialTraceExtendedTests(Question):
    """
    Extended partial trace tests: qutrit systems, keep='A' vs keep='B'.
    """

    def test_trace_product_keep_B(self):
        """
        $\\mathrm{Tr}_A(|01\\rangle\\langle01|) = |1\\rangle\\langle1|$: tracing out A gives B state
        """

        # |01> state: index 1 in 4-dim space
        ket01 = np.zeros(4, dtype=complex)
        ket01[1] = 1.0
        rho = np.outer(ket01, ket01.conj())

        rho_B = np.array(partial.trace(rho, 2, 2, keep="B"))

        self.assertAlmostEqual(
            float(np.abs(rho_B[1, 1])),
            1.0,
            places=6,
            msg="Tr_A(|01><01|) should give |1><1|",

        )

        self.assertAlmostEqual(float(np.abs(rho_B[0, 0])), 0.0, places=6, msg="Tr_A(|01><01|)[0,0] should be 0")

    def test_trace_entangled_both_subsystems(self):
        """

        For Bell state, both $\\mathrm{Tr}_A$ and $\\mathrm{Tr}_B$ give $I/2$
        """
        bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        rho = np.outer(bell, bell.conj())
        rho_A = np.array(partial.trace(rho, 2, 2, keep="A"))
        rho_B = np.array(partial.trace(rho, 2, 2, keep="B"))
        for sub, label in [(rho_A, "A"), (rho_B, "B")]:
            self.assertAlmostEqual(
                float(np.abs(sub[0, 0])),
                0.5,
                places=5,
                msg=f"Tr_{label}(Bell)[0,0] should be 0.5",

            )

            self.assertAlmostEqual(
                float(np.abs(sub[1, 1])),
                0.5,
                places=5,
                msg=f"Tr_{label}(Bell)[1,1] should be 0.5",
            )


    def test_transpose_negative_eigenvalue(self):
        """
        Partial transpose of Bell state has negative eigenvalue $-1/2$:
        $(\\rho^{T_B})$ has spectrum $\\{1/2, 1/2, 1/2, -1/2\\}$
        """
        bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        rho = np.outer(bell, bell.conj())
        rho_pt = np.array(partial.transpose(rho, 2, 2))

        evals = np.sort(np.linalg.eigvalsh(rho_pt))

        self.assertAlmostEqual(
            float(evals[0]),
            -0.5,

            places=5,
            msg="Bell state partial transpose should have eigenvalue -1/2",
        )


class BraketExtendedTests(Question):
    """
    Extended Braket and Tensor tests with operators.
    """

    def test_braket_with_operator(self):
        """
        $\\langle 0 | Z | 0 \\rangle = 1$: expectation of $Z$ in $|0\\rangle$ is $+1$

        """
        B = Basis(2)
        Z = pt.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=pt.complex128)
        ket0 = pt.as_tensor(B(0), dtype=pt.complex128)

        result = Braket(ket0, Z, ket0)

        self.assertAlmostEqual(
            float(pt.real(result).item()), 1.0, places=6, msg="<0|Z|0> should be 1"
        )

    def test_braket_z_excited(self):
        """
        $\\langle 1 | Z | 1 \\rangle = -1$: expectation of $Z$ in $|1\\rangle$ is $-1$

        """
        B = Basis(2)
        Z = pt.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=pt.complex128)
        ket1 = pt.as_tensor(B(1), dtype=pt.complex128)

        result = Braket(ket1, Z, ket1)

        self.assertAlmostEqual(
            float(pt.real(result).item()), -1.0, places=6, msg="<1|Z|1> should be -1"
        )

    def test_braket_x_superposition(self):
        """
        $\\langle + | X | + \\rangle = 1$: $X$ expectation in $|{+}\\rangle$ is $+1$
        """
        B = Basis(2)

        X = pt.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=pt.complex128)
        plus = pt.as_tensor((B(0) + B(1)) / np.sqrt(2), dtype=pt.complex128)

        result = Braket(plus, X, plus)

        self.assertAlmostEqual(
            float(pt.real(result).item()), 1.0, places=5, msg="<+|X|+> should be 1"
        )

    def test_tensor_three_factors(self):
        """
        $|0\\rangle \\otimes |1\\rangle \\otimes |0\\rangle$ has amplitude 1 at index 2
        (binary: 010 = 2)
        """
        B = Basis(2)
        ket010 = Tensor(B(0), B(1), B(0))

        arr = np.array(ket010)

        self.assertAlmostEqual(
            float(np.abs(arr[2])),
            1.0,
            places=6,
            msg="|010> should have amplitude 1 at index 2",
        )

    def test_tensor_associativity(self):
        """
        $(|0\\rangle \\otimes |1\\rangle) \\otimes |0\\rangle = |0\\rangle \\otimes (|1\\rangle \\otimes |0\\rangle)$
        """
        B = Basis(2)
        left = Tensor(Tensor(B(0), B(1)), B(0))

        right = Tensor(B(0), Tensor(B(1), B(0)))

        np.testing.assert_allclose(np.array(left), np.array(right), atol=1e-10)


if __name__ == "__main__":
    runner = Exam(
        name="Tools Extra Tests",
        desc="Space, partial trace, Braket/Tensor utility tests",
        file="tools_extra.md",
    )
    runner.run(load(SpaceTests))
    runner.run(load(PartialTraceTests))
    runner.run(load(BraketTensorTests))
    runner.run(load(SpaceExtendedTests))
    runner.run(load(PartialTraceExtendedTests))
    runner.run(load(BraketExtendedTests))
