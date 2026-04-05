from MDR import Exam, load, Question
import sys
import torch as pt

sys.path.append("..")

from qudit.noise import Process
from qudit.qec import Recovery
from qudit.qec.lib import Leung, Dutta3, Perfect, Surface, Qutrit3, GottesmanD
from qudit.qec.codes import Code


def to_rho(x):
    size = x.numel()

    return (x.view(size, 1) @ pt.conj(x.view(1, size))).to(pt.complex64)


fid = lambda rho, sigma: pt.real(pt.trace(rho @ sigma)).item()


class QEC(Question):
    """
    Quantum error-correction tests: code orthonormality and
    recovery map fidelity on noisy codewords.
    """

    def test_leung_orthonormal(self):
        """
        Leung codewords are orthonormal:
        $\\langle 0_L | 0_L \\rangle = \\langle 1_L | 1_L \\rangle = 1$,
        $\\langle 0_L | 1_L \\rangle = 0$
        """
        state0, state1 = Leung().toTensor()
        inner_00 = pt.dot(state0, state0).real.item()
        inner_11 = pt.dot(state1, state1).real.item()

        inner_01 = pt.abs(pt.dot(state0, state1)).item()

        self.assertAlmostEqual(inner_00, 1.0, places=6, msg="Leung |0_L> not normalized")

        self.assertAlmostEqual(inner_11, 1.0, places=6, msg="Leung |1_L> not normalized")

        self.assertAlmostEqual(inner_01, 0.0, places=6, msg="Leung codewords not orthogonal")

    def test_petz_ad(self):
        """
        $\\mathcal{R}_\\mathrm{Petz}$ on amplitude-damping noise ($\\gamma=0.1$,
        order 3) with Leung 4-qubit codewords recovers fidelity $\\approx 0.9889$
        and strictly improves on the noisy fidelity
        """
        n = 4
        state0, state1 = Leung().toTensor()
        rho0, rho1 = to_rho(state0), to_rho(state1)

        noise = Process.AD(d=2, n=n, Y=0.1, order=3)

        noisy0 = noise.run(rho0)
        noisy1 = noise.run(rho1)

        rec = Recovery.petz(noise, [state0, state1])

        clean0 = rec.run(noisy0)
        clean1 = rec.run(noisy1)

        fid0_noisy = fid(rho0, noisy0)
        fid0_clean = fid(rho0, clean0)

        fid1_noisy = fid(rho1, noisy1)
        fid1_clean = fid(rho1, clean1)

        self.assertAlmostEqual(
            fid0_clean,
            0.9889,
            places=2,
            msg="Petz AD recovery fidelity mismatch for codeword 0",

        )

        self.assertAlmostEqual(
            fid1_clean,
            0.9889,
            places=2,
            msg="Petz AD recovery fidelity mismatch for codeword 1",

        )

        self.assertGreater(
            fid0_clean,
            fid0_noisy,
            msg="Recovery should improve fidelity for codeword 0",

        )

        self.assertGreater(
            fid1_clean,
            fid1_noisy,
            msg="Recovery should improve fidelity for codeword 1",
        )


class QECCodes(Question):
    """
    Code orthonormality tests for Dutta3, Perfect, and stabilizer-derived codes.
    """

    def _check_orthonormal(self, code, name):
        cw = code.toTensor()
        inner_00 = pt.dot(cw[0], cw[0]).real.item()
        inner_11 = pt.dot(cw[1], cw[1]).real.item()

        inner_01 = pt.abs(pt.dot(cw[0], cw[1])).item()

        self.assertAlmostEqual(
            inner_00, 1.0, places=5, msg=f"{name}: |0_L> not normalized"

        )

        self.assertAlmostEqual(
            inner_11, 1.0, places=5, msg=f"{name}: |1_L> not normalized"

        )

        self.assertAlmostEqual(
            inner_01, 0.0, places=5, msg=f"{name}: codewords not orthogonal"
        )

    def test_dutta3_orthonormal(self):
        """
        Dutta3 codewords are orthonormal:
        $\\langle 0_L | 0_L \\rangle = \\langle 1_L | 1_L \\rangle = 1$,
        $\\langle 0_L | 1_L \\rangle = 0$
        """
        self._check_orthonormal(Dutta3(), "Dutta3")

    def test_dutta3_structure(self):
        """
        $|0_L\\rangle = (|001\\rangle + |010\\rangle + |100\\rangle)/\\sqrt{3}$,
        $|1_L\\rangle = |111\\rangle$: correct Dutta3 codeword structure
        """
        code = Dutta3()
        cw = code.toTensor()
        # |0_L> has support at indices 1,2,4 with amplitude 1/√3

        amp0 = 1.0 / pt.sqrt(pt.tensor(3.0))

        self.assertAlmostEqual(float(cw[0, 1]), float(amp0), places=5, msg="Dutta3 |0_L> amplitude at index 1")

        self.assertAlmostEqual(float(cw[0, 2]), float(amp0), places=5, msg="Dutta3 |0_L> amplitude at index 2")

        self.assertAlmostEqual(float(cw[0, 4]), float(amp0), places=5, msg="Dutta3 |0_L> amplitude at index 4")
        # |1_L> has support only at index 7 (|111>) with amplitude 1
        self.assertAlmostEqual(float(cw[1, 7]), 1.0, places=5, msg="Dutta3 |1_L> amplitude at index 7")

    def test_perfect_orthonormal(self):
        """
        [[5,1,3]] Perfect code codewords are orthonormal:
        $\\langle 0_L | 0_L \\rangle = \\langle 1_L | 1_L \\rangle = 1$,
        $\\langle 0_L | 1_L \\rangle = 0$
        """
        self._check_orthonormal(Perfect(), "Perfect[[5,1,3]]")

    def test_perfect_codeword_count(self):
        """
        Perfect code has 2 codewords spanning $\\mathbb{C}^{32}$ (5-qubit space)
        """
        code = Perfect()

        cw = code.toTensor()

        self.assertEqual(cw.shape[0], 2, "Perfect code should have 2 codewords")

        self.assertEqual(cw.shape[1], 32, "Perfect code codewords in 2^5=32 dim space")

    def test_from_stabilizers_3qubit_bitflip(self):
        """
        3-qubit bit-flip code $\\{ZZI, IZZ\\}$: $Code.fromStabilizers$ produces
        2 orthonormal codewords from the $+1$ eigenspace
        """
        code = Code.fromStabilizers(["ZZI", "IZZ"])

        cw = code.toTensor()

        self.assertEqual(
            cw.shape[0], 2, "3-qubit bit-flip code should have 2 codewords"
        )
        # Both codewords should be normalized
        for i in range(2):

            norm = float(pt.linalg.norm(cw[i]).item())

            self.assertAlmostEqual(
                norm,
                1.0,
                places=4,
                msg=f"Codeword {i} of 3-qubit flip code not normalized",
            )
        # Codewords should be orthogonal

        inner = pt.abs(pt.dot(cw[0], cw[1])).item()

        self.assertAlmostEqual(
            inner, 0.0, places=4, msg="3-qubit flip codewords should be orthogonal"
        )

    def test_from_stabilizers_codewords_are_eigenstates(self):
        """
        Codewords from $\\{ZZI, IZZ\\}$ are $+1$ eigenstates of all stabilizers:
        $S|\\psi\\rangle = |\\psi\\rangle$ for each generator $S$
        """
        import torch as pt

        Pauli = {
            "I": pt.tensor([[1, 0], [0, 1]], dtype=pt.complex64),
            "Z": pt.tensor([[1, 0], [0, -1]], dtype=pt.complex64),
        }

        def composite(s):
            mat = Pauli[s[0]]
            for p in s[1:]:
                mat = pt.kron(mat, Pauli[p])

            return mat

        stabs = ["ZZI", "IZZ"]
        code = Code.fromStabilizers(stabs)
        cw = code.toTensor().to(pt.complex64)

        for stab in stabs:
            S = composite(list(stab))

            for i, v in enumerate(cw):
                diff = float(pt.norm(S @ v - v).item())

                self.assertAlmostEqual(
                    diff, 0.0, places=3, msg=f"Codeword {i} not +1 eigenstate of {stab}"
                )

    def test_from_stabilizers_svd_vs_rrf(self):
        """
        SVD and RRF methods of $Code.fromStabilizers$ produce orthonormal codespaces
        for the 3-qubit bit-flip code
        """
        code_svd = Code.fromStabilizers(["ZZI", "IZZ"], method="svd")
        code_rrf = Code.fromStabilizers(["ZZI", "IZZ"], method="rrf")
        # Both should produce normalized codewords
        for code, label in [(code_svd, "SVD"), (code_rrf, "RRF")]:
            cw = code.toTensor()

            for i in range(len(code)):
                norm = float(pt.linalg.norm(cw[i]).item())

                self.assertAlmostEqual(
                    norm, 1.0, places=3, msg=f"{label} codeword {i} not normalized"
                )

    def test_from_stabilizers_qutrit(self):
        """
        $Code.fromStabilizers$ with $d=3$: 2 qutrit stabilizers on 2 sites yield
        $3^{2-2}=1$ codeword in a $3^2=9$-dimensional space
        """

        code = Code.fromStabilizers(["ZI", "IZ"], d=3)

        cw = code.toTensor()

        self.assertEqual(cw.shape[0], 1, "Expected 1 qutrit codeword")

        self.assertEqual(cw.shape[1], 9, "Qutrit 2-site space should be 9-dimensional")
        norm = float(pt.linalg.norm(cw[0]).item())

        self.assertAlmostEqual(
            norm, 1.0, places=4, msg="Qutrit codeword not normalized"
        )

    def test_surface_d2(self):
        """
        $Surface(3, 1, d=2)$: 1×3 qubit surface code yields $2^{3-2}=2$ codewords
        in an $2^3=8$-dimensional space
        """

        code = Surface(3, 1, d=2)
        cw = code.toTensor()

        self.assertEqual(

            cw.shape[0], 2, "1×3 qubit surface code should have 2 codewords"
        )

        self.assertEqual(cw.shape[1], 8, "Qubit 3-site space should be 8-dimensional")

    def test_surface_d3(self):
        """
        $Surface(3, 1, d=3)$: 1×3 qutrit surface code yields $3^{3-2}=3$ codewords
        in a $3^3=27$-dimensional space
        """

        code = Surface(3, 1, d=3)
        cw = code.toTensor()

        self.assertEqual(

            cw.shape[0], 3, "1×3 qutrit surface code should have 3 codewords"
        )

        self.assertEqual(
            cw.shape[1], 27, "Qutrit 3-site space should be 27-dimensional"
        )


class QuditCodes(Question):
    """
    Qutrit and higher-d built-in codes: orthonormality and shape.
    """

    def test_qutrit3_orthonormal(self):
        """
        $Qutrit3$ codewords are orthonormal: 3 codewords in $3^3=27$-dimensional space
        """

        code = Qutrit3()
        cw = code.toTensor()

        self.assertEqual(cw.shape, pt.Size([3, 27]), msg="Qutrit3 shape should be [3, 27]")
        gram = (cw @ cw.T).real

        self.assertTrue(
            pt.allclose(gram, pt.eye(3), atol=1e-5),
            msg="Qutrit3 codewords not orthonormal",
        )

    def test_qutrit3_codeword_structure(self):
        """
        $Qutrit3$: $|0_L\\rangle=|000\\rangle$, $|1_L\\rangle=|111\\rangle$, $|2_L\\rangle=|222\\rangle$

        """

        cw = Qutrit3().toTensor()

        self.assertAlmostEqual(float(cw[0, 0]), 1.0, places=5, msg="Qutrit3 |0_L>=|000> amplitude mismatch")

        self.assertAlmostEqual(float(cw[1, 13]), 1.0, places=5, msg="Qutrit3 |1_L>=|111> amplitude mismatch")

        self.assertAlmostEqual(float(cw[2, 26]), 1.0, places=5, msg="Qutrit3 |2_L>=|222> amplitude mismatch")

    def test_gottesman_d2_orthonormal(self):
        """
        $GottesmanD(d=2)$: 2 codewords in $2^2=4$-dimensional space, orthonormal
        """

        code = GottesmanD(d=2)

        cw = code.toTensor()

        self.assertEqual(cw.shape[0], 2, msg="GottesmanD(d=2) should have 2 codewords")

        self.assertEqual(cw.shape[1], 4, msg="GottesmanD(d=2) codewords in 4-dim space")

        for i in range(2):
            norm = float(pt.linalg.norm(cw[i]).item())

            self.assertAlmostEqual(norm, 1.0, places=4, msg=f"GottesmanD(d=2) codeword {i} not normalized")
        inner = pt.abs(cw[0].conj() @ cw[1]).item()

        self.assertAlmostEqual(inner, 0.0, places=4, msg="GottesmanD(d=2) codewords not orthogonal")

    def test_gottesman_d3_orthonormal(self):
        """
        $GottesmanD(d=3)$: 3 codewords in $3^3=27$-dimensional space, orthonormal
        """

        code = GottesmanD(d=3)
        cw = code.toTensor()

        self.assertEqual(cw.shape, pt.Size([3, 27]), msg="GottesmanD(d=3) shape should be [3, 27]")
        gram = (cw @ cw.conj().T).real

        self.assertTrue(
            pt.allclose(gram, pt.eye(3), atol=1e-5),
            msg="GottesmanD(3) codewords not orthonormal",
        )


if __name__ == "__main__":
    runner = Exam(
        name="Qudit ECC Tests",
        desc="Validation of error-correction code properties and recovery maps",
        file="ECC.md",
    )
    runner.run(load(QEC))
    runner.run(load(QECCodes))
    runner.run(load(QuditCodes))
