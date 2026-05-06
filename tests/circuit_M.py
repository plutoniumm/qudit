from MDR import Exam, load, Question
import sys

sys.path.append("..")

from qudit import Circuit, Mode
from qudit.noise import WeylNoise, PhysicalNoise, CoherentNoise
import torch

C64 = torch.complex64


def ket0(size):
    x = torch.zeros(size, dtype=C64)
    x[0] = 1.0

    return x


def toRho(x):
    size = x.numel()

    return x.view(size, 1) @ torch.conj(x.view(1, size))


class MatrixCircuit(Question):
    """
    Circuit tests in density-matrix (MATRIX) mode, verifying
    $\\Phi(|\\psi\\rangle\\langle\\psi|) = U|\\psi\\rangle\\langle\\psi|U^\\dagger$.
    """

    def test_single_qubit(self):
        """
        $H: U|\\psi\\rangle\\langle\\psi|U^\\dagger = \\Phi(|\\psi\\rangle\\langle\\psi|)$
        for a single qubit, two ways
        """
        cM = Circuit(wires=1, dim=2, mode=Mode.MATRIX)
        cV = Circuit(wires=1, dim=2, mode=Mode.VECTOR)
        G2 = cM.gates[2]

        cM.gate(G2.H, [0])
        cV.gate(G2.H, [0])

        xV = ket0(cM.width)
        xM = toRho(xV)

        rhoV = toRho(cV(xV))
        rhoM = cM(xM)

        self.matEqual(
            rhoV, rhoM, msg="Single-qubit: VECTOR and MATRIX modes should agree"
        )

    def test_single_qutrit(self):
        """
        $H: U|\\psi\\rangle\\langle\\psi|U^\\dagger = \\Phi(|\\psi\\rangle\\langle\\psi|)$
        for a single qutrit, two ways
        """
        cM = Circuit(wires=1, dim=3, mode=Mode.MATRIX)
        cV = Circuit(wires=1, dim=3, mode=Mode.VECTOR)
        G3 = cM.gates[3]

        cM.gate(G3.H, [0])
        cV.gate(G3.H, [0])

        xV = ket0(cM.width)
        xM = toRho(xV)

        rhoV = toRho(cV(xV))
        rhoM = cM(xM)

        self.matEqual(
            rhoV, rhoM, msg="Single-qutrit: VECTOR and MATRIX modes should agree"
        )

    def test_mixed_dims(self):
        """
        Mixed-dimension $[2,2,3,3]$ circuit: VECTOR and MATRIX modes agree
        under $H, X, CX$, two ways
        """
        cM = Circuit(wires=4, dim=[2, 2, 3, 3], mode=Mode.MATRIX)
        cV = Circuit(wires=4, dim=[2, 2, 3, 3], mode=Mode.VECTOR)
        G2 = cM.gates[2]
        G3 = cM.gates[3]

        cM.gate(G2.H, [0])
        cM.gate(G2.X, [1])
        cM.gate(G3.CX, [2, 3])

        cV.gate(G2.H, [0])
        cV.gate(G2.X, [1])
        cV.gate(G3.CX, [2, 3])

        xV = ket0(cM.width)
        xM = toRho(xV)

        rhoV = toRho(cV(xV))
        rhoM = cM(xM)

        self.matEqual(
            rhoV, rhoM, msg="Mixed-dims [2,2,3,3]: VECTOR and MATRIX modes should agree"
        )

    def test_trace_preserved(self):
        """
        $\\mathrm{Tr}(U\\rho U^\\dagger) = \\mathrm{Tr}(\\rho) = 1$:
        unitary channels are trace-preserving
        """
        cM = Circuit(wires=2, dim=2, mode=Mode.MATRIX)
        G2 = cM.gates[2]
        cM.gate(G2.H, [0])
        cM.gate(G2.CX, [0, 1])

        xV = ket0(cM.width)

        rhoM = cM(toRho(xV))
        tr = torch.trace(rhoM).real.item()

        self.assertAlmostEqual(
            tr, 1.0, places=5, msg="Unitary channel should preserve trace"
        )

    def test_purity_preserved(self):
        """
        $\\mathrm{Tr}((U\\rho U^\\dagger)^2) = \\mathrm{Tr}(\\rho^2) = 1$:
        unitary channels preserve purity
        """
        cM = Circuit(wires=2, dim=2, mode=Mode.MATRIX)
        G2 = cM.gates[2]
        cM.gate(G2.H, [0])
        cM.gate(G2.CX, [0, 1])

        xV = ket0(cM.width)

        rhoM = cM(toRho(xV))
        purity = torch.trace(rhoM @ rhoM).real.item()

        self.assertAlmostEqual(
            purity, 1.0, places=5, msg="Unitary channel should preserve purity"
        )


class NoisyCircuit(Question):
    """
    Circuit tests in NOISY mode (WeylNoise per gate): trace preservation,
    Hermiticity, PSD, and deviation from the noiseless case.
    """

    def _noisy(self, p):
        c = Circuit(wires=1, dim=2, mode=Mode.NOISY, noise=WeylNoise(p, seed=0))
        c.gate(c.gates[2].H, [0])

        return c

    def test_noisy_trace_preserved(self):
        """
        NOISY mode preserves trace: $\\mathrm{Tr}(\\Phi(\\rho)) = 1$
        """
        c = self._noisy(0.05)
        rho = toRho(ket0(c.width))
        out = c(rho)

        self.assertAlmostEqual(
            torch.trace(out).real.item(),
            1.0,
            places=4,
            msg="Noisy circuit should preserve trace",
        )

    def test_noisy_hermitian(self):
        """
        NOISY mode output is Hermitian: $\\rho^\\dagger = \\rho$
        """
        c = self._noisy(0.05)
        rho = toRho(ket0(c.width))
        out = c(rho)
        diff = torch.norm(out - out.conj().T).real.item()

        self.assertAlmostEqual(
            diff, 0.0, places=4, msg="Noisy circuit output should be Hermitian"
        )

    def test_noisy_psd(self):
        """
        NOISY mode output is positive semidefinite: all eigenvalues $\\geq 0$
        """
        c = self._noisy(0.05)
        rho = toRho(ket0(c.width))
        out = c(rho)
        evals = torch.linalg.eigvalsh(out.real)

        self.assertTrue(
            bool((evals >= -1e-5).all().item()),
            msg="Noisy circuit output should be positive semidefinite",
        )

    def test_noisy_differs_from_noiseless(self):
        """
        NOISY averaged over shots differs from noiseless MATRIX output (noise has visible effect)
        """
        cN = self._noisy(0.9)
        cM = Circuit(wires=1, dim=2, mode=Mode.MATRIX)
        cM.gate(cM.gates[2].H, [0])

        rho = toRho(ket0(2))
        out_clean = cM(rho)
        rho_avg = sum(cN(rho) for _ in range(50)) / 50
        diff = torch.norm(rho_avg - out_clean).item()

        self.assertGreater(
            diff,
            1e-3,
            msg="Noisy circuit (averaged) should differ from noiseless for large noise",
        )

    def test_seeded_reproducible(self):
        """
        NOISY mode with fixed seed produces the same output across two runs
        """

        def run():
            c = Circuit(wires=1, dim=2, mode=Mode.NOISY, noise=WeylNoise(0.1, seed=42))
            c.gate(c.gates[2].H, [0])

            return c(toRho(ket0(2)))

        out1 = run()
        out2 = run()
        diff = torch.norm(out1 - out2).item()

        self.assertAlmostEqual(
            diff, 0.0, places=5, msg="Seeded NOISY circuit should be reproducible"
        )


class TrajectoryCircuit(Question):
    """
    Circuit tests in TRAJECTORY mode (quantum jump statevector simulation).
    """

    def _traj(self, p, seed=0):
        c = Circuit(wires=1, dim=2, mode=Mode.TRAJECTORY, noise=WeylNoise(p, seed=seed))
        c.gate(c.gates[2].H, [0])

        return c

    def test_trajectory_normalized(self):
        """
        TRAJECTORY output is a normalized statevector: $\\|\\psi\\|^2 = 1$
        """
        c = self._traj(0.05)
        psi = c(ket0(2))
        norm = psi.norm().real.item()

        self.assertAlmostEqual(
            norm, 1.0, places=4, msg="Trajectory output should be normalized"
        )

    def test_trajectory_seeded_reproducible(self):
        """
        TRAJECTORY mode with fixed seed reproduces the same statevector
        """

        def run():
            c = self._traj(0.1, seed=7)

            return c(ket0(2))

        out1 = run()
        out2 = run()
        diff = torch.norm(out1 - out2).item()

        self.assertAlmostEqual(
            diff, 0.0, places=5, msg="Seeded TRAJECTORY should be reproducible"
        )

    def test_physical_noise_trajectory(self):
        """
        PhysicalNoise in TRAJECTORY mode yields normalized statevector
        """
        c = Circuit(
            wires=1,
            dim=2,
            mode=Mode.TRAJECTORY,
            noise=PhysicalNoise(T1=50e-6, T2=30e-6, seed=0),
        )
        c.gate(c.gates[2].H, [0])
        psi = c(ket0(2))
        norm = psi.norm().real.item()

        self.assertAlmostEqual(
            norm,
            1.0,
            places=4,
            msg="PhysicalNoise TRAJECTORY output should be normalized",
        )


class CircuitMatrix(Question):
    """
    Tests for `circuit.matrix()`: unitarity, known matrix values,
    and consistency with the forward pass on basis vectors.
    """

    def test_H_unitary(self):
        """
        $H$ on 1 qubit: $U^\\dagger U = I$
        """
        c = Circuit(wires=1, dim=2)
        c.gate(c.gates[2].H, [0])

        M = c.matrix()
        prod = M.conj().T @ M
        I = torch.eye(2, dtype=C64)

        self.matEqual(prod, I, msg="H gate matrix should be unitary (U†U = I)")

    def test_CX_known_matrix(self):
        """
        $CX$ on 2 qubits: matrix matches the known $4\\times4$ CNOT matrix
        """
        c = Circuit(wires=2, dim=2)
        c.gate(c.gates[2].CX, [0, 1])

        M = c.matrix()
        expected = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=C64,
        )

        self.matEqual(M, expected, msg="CX matrix should match known 4×4 CNOT matrix")

    def test_matrix_columns_eq_forward(self):
        """
        Each column of `c.matrix()` equals the forward pass on the corresponding basis vector.
        """
        c = Circuit(wires=2, dim=2)
        G2 = c.gates[2]
        c.gate(G2.H, [0])
        c.gate(G2.CX, [0, 1])

        M = c.matrix()
        W = c.width

        for i in range(W):
            ei = torch.zeros(W, dtype=C64)
            ei[i] = 1.0
            col = c(ei).reshape(-1)

            self.matEqual(
                M[:, i],
                col,
                msg=f"Column {i} of matrix() should equal forward pass on basis vector |{i}>",
            )


class CircuitSample(Question):
    """
    Tests for `circuit.sample()`: distribution correctness, key format, and count totals.
    """

    def test_H_approx_uniform(self):
        """
        $H|0\\rangle$ sampled 1000 times: "0" and "1" each appear within 10% of 50/50
        """
        torch.manual_seed(0)
        c = Circuit(wires=1, dim=2)
        c.gate(c.gates[2].H, [0])

        x = ket0(c.width)
        counts = c.sample(x, shots=1000)

        p0 = counts.get("0", 0) / 1000
        p1 = counts.get("1", 0) / 1000

        self.assertAlmostEqual(
            p0, 0.5, delta=0.1, msg="H|0> should give ~50% '0' outcomes"
        )

        self.assertAlmostEqual(
            p1, 0.5, delta=0.1, msg="H|0> should give ~50% '1' outcomes"
        )

    def test_bell_dominant_outcomes(self):
        """
        Bell state $|\\Phi^+\\rangle$ sampled 1000 times: "00" + "11" total probability > 0.9
        """
        torch.manual_seed(0)
        c = Circuit(wires=2, dim=2)
        G2 = c.gates[2]
        c.gate(G2.H, [0])
        c.gate(G2.CX, [0, 1])

        x = ket0(c.width)
        counts = c.sample(x, shots=1000)

        dominant = (counts.get("00", 0) + counts.get("11", 0)) / 1000

        self.assertGreater(
            dominant, 0.9, msg="Bell state should give >90% '00'+'11' outcomes"
        )

    def test_sample_string_keys(self):
        """
        `sample()` returns a dict with string keys
        """
        c = Circuit(wires=2, dim=2)
        c.gate(c.gates[2].H, [0])

        counts = c.sample(ket0(c.width), shots=100)

        for k in counts:
            self.assertIsInstance(k, str, msg=f"Sample key {k!r} should be a string")

    def test_sample_counts_sum_to_shots(self):
        """
        Sum of all sample counts equals `shots`
        """
        shots = 500
        c = Circuit(wires=2, dim=2)
        G2 = c.gates[2]
        c.gate(G2.H, [0])
        c.gate(G2.CX, [0, 1])

        counts = c.sample(ket0(c.width), shots=shots)
        total = sum(counts.values())

        self.assertEqual(total, shots, msg=f"Sample counts should sum to shots={shots}")


class CircuitExpectation(Question):
    """
    Tests for `circuit.expectation()`: known eigenvalue results and VECTOR/MATRIX agreement.
    """

    Z = torch.tensor([[1, 0], [0, -1]], dtype=C64)

    def test_Z_on_ket0(self):
        """
        $\\langle 0|Z|0\\rangle = +1$
        """
        c = Circuit(wires=1, dim=2)
        c.gate(c.gates[2].I, [0])

        val = c.expectation(self.Z, ket0(2)).item()

        self.assertAlmostEqual(
            val, 1.0, places=5, msg="Z expectation on |0> should be +1"
        )

    def test_Z_on_ket1(self):
        """
        $\\langle 1|Z|1\\rangle = -1$ after applying $X|0\\rangle = |1\\rangle$
        """
        c = Circuit(wires=1, dim=2)
        c.gate(c.gates[2].X, [0])

        val = c.expectation(self.Z, ket0(2)).item()

        self.assertAlmostEqual(
            val, -1.0, places=5, msg="Z expectation on |1> should be -1"
        )

    def test_ZI_on_bell(self):
        """
        $\\langle\\Phi^+|Z\\otimes I|\\Phi^+\\rangle = 0$
        """
        c = Circuit(wires=2, dim=2)
        G2 = c.gates[2]
        c.gate(G2.H, [0])
        c.gate(G2.CX, [0, 1])

        ZI = torch.kron(self.Z, torch.eye(2, dtype=C64))
        val = c.expectation(ZI, ket0(4)).item()

        self.assertAlmostEqual(
            val, 0.0, places=5, msg="Z⊗I expectation on Bell state should be 0"
        )

    def test_matrix_mode_agrees_with_vector(self):
        """
        MATRIX mode expectation matches VECTOR mode: $\\mathrm{Tr}(Z\\rho) = \\langle\\psi|Z|\\psi\\rangle$
        """
        cV = Circuit(wires=1, dim=2, mode=Mode.VECTOR)
        cM = Circuit(wires=1, dim=2, mode=Mode.MATRIX)
        G2V = cV.gates[2]
        G2M = cM.gates[2]
        cV.gate(G2V.H, [0])
        cM.gate(G2M.H, [0])

        x = ket0(2)
        rho = toRho(x)

        valV = cV.expectation(self.Z, x).item()
        valM = cM.expectation(self.Z, rho).item()

        self.assertAlmostEqual(
            valV,
            valM,
            places=5,
            msg="MATRIX and VECTOR mode should give the same Z expectation",
        )


class LargerCircuits(Question):
    """
    Tests for 3- and 4-qubit circuits and mixed-dimension circuits.
    """

    def test_ghz_vector_matrix_agree(self):
        """
        3-qubit GHZ circuit ($H_0$, $CX_{01}$, $CX_{12}$): VECTOR and MATRIX modes agree
        """
        cV = Circuit(wires=3, dim=2, mode=Mode.VECTOR)
        cM = Circuit(wires=3, dim=2, mode=Mode.MATRIX)

        for c in (cV, cM):
            G = c.gates[2]
            c.gate(G.H, [0])
            c.gate(G.CX, [0, 1])
            c.gate(G.CX, [1, 2])

        xV = ket0(cV.width)
        xM = toRho(xV)

        rhoV = toRho(cV(xV))
        rhoM = cM(xM)

        self.matEqual(
            rhoV, rhoM, msg="3-qubit GHZ: VECTOR and MATRIX modes should agree"
        )

    def test_4qubit_trace_preserved(self):
        """
        4-qubit unitary circuit ($H$, $CX$, $CX$, $CX$): trace equals 1 after channel
        """
        c = Circuit(wires=4, dim=2, mode=Mode.MATRIX)
        G = c.gates[2]
        c.gate(G.H, [0])
        c.gate(G.CX, [0, 1])
        c.gate(G.CX, [1, 2])
        c.gate(G.CX, [2, 3])

        rho = toRho(ket0(c.width))
        rho_out = c(rho)
        tr = torch.trace(rho_out).real.item()

        self.assertAlmostEqual(
            tr, 1.0, places=4, msg="4-qubit unitary channel should preserve trace"
        )

    def test_mixed_dim_23_agree(self):
        """
        Mixed-dimension $[2,3]$ 2-wire circuit ($H_0$, $X_1$): VECTOR and MATRIX modes agree
        """
        cV = Circuit(wires=2, dim=[2, 3], mode=Mode.VECTOR)
        cM = Circuit(wires=2, dim=[2, 3], mode=Mode.MATRIX)

        for c in (cV, cM):
            c.gate(c.gates[2].H, [0])
            c.gate(c.gates[3].X, [1])

        xV = ket0(cV.width)
        xM = toRho(xV)

        rhoV = toRho(cV(xV))
        rhoM = cM(xM)

        self.matEqual(
            rhoV, rhoM, msg="Mixed-dim [2,3]: VECTOR and MATRIX modes should agree"
        )


if __name__ == "__main__":
    runner = Exam(
        name="Qudit Matrix Circuit Tests",
        desc="Validation of circuit forward pass in density-matrix mode",
        file="circuit_m.md",
    )
    runner.run(load(MatrixCircuit))
    runner.run(load(NoisyCircuit))
    runner.run(load(TrajectoryCircuit))
    runner.run(load(CircuitMatrix))
    runner.run(load(CircuitSample))
    runner.run(load(CircuitExpectation))
    runner.run(load(LargerCircuits))
