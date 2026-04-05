from MDR import Exam, load, Question
import sys

sys.path.append("..")

from qudit import Circuit, Basis, State
import torch

C64 = torch.complex64
dev = "cpu"
Ket2 = Basis(2)
Ket3 = Basis(3)


def ket0(size):
    x = torch.zeros(size, dtype=C64, device=dev)
    x[0] = 1.0

    return x


class VectorCircuit(Question):
    """
    Circuit forward-pass tests in statevector (VECTOR) mode for qubits, qutrits,
    and mixed-dimension systems.
    """

    def test_bell_state(self):
        """
        $H \\otimes I \\cdot CX|00\\rangle = \\frac{1}{\\sqrt{2}}(|00\\rangle + |11\\rangle)$
        """
        c = Circuit(wires=2, dim=2, device=dev)
        G = c.gates[2]
        c.gate(G.H, [0])
        c.gate(G.CX, [0, 1])

        psi = c(ket0(c.width))

        exp = State(Ket2(0, 0) + Ket2(1, 1))

        self.stateEqual(exp, psi, msg="Bell state mismatch")

    def test_mixed_dimension_ent(self):
        """
        Mixed-dimension $[2,2,3,3]$ circuit produces
        $\\frac{1}{\\sqrt{2}}(|0100\\rangle + |1100\\rangle)$
        """
        c = Circuit(wires=4, dim=[2, 2, 3, 3], device=dev)
        G2 = c.gates[2]
        G3 = c.gates[3]
        c.gate(G2.H, [0])
        c.gate(G2.X, [1])
        c.gate(G3.CX, [2, 3])

        psi = c(ket0(c.width))
        exp = State(
            (Ket2(0) ^ Ket2(1) ^ Ket3(0) ^ Ket3(0))
            + (Ket2(1) ^ Ket2(1) ^ Ket3(0) ^ Ket3(0))

        )

        self.stateEqual(exp, psi, msg="Mixed-dim entangled state mismatch")

    def test_three_qutrit_ghz(self):
        """
        $\\frac{1}{\\sqrt{3}}(|000\\rangle + |111\\rangle + |222\\rangle)$
        via qutrit GHZ circuit
        """
        c = Circuit(wires=3, dim=3, device=dev)
        G3 = c.gates[3]
        c.gate(G3.H, [0])
        c.gate(G3.CX, [0, 1])
        c.gate(G3.CX, [0, 2])

        psi = c(ket0(c.width))

        exp = State(Ket3(0, 0, 0) + Ket3(1, 1, 1) + Ket3(2, 2, 2))

        self.stateEqual(exp, psi, msg="Qutrit GHZ state mismatch")

    def test_h_involution(self):
        """
        $H^2 = I$: applying $H$ twice returns the original state, two ways
        """
        c = Circuit(wires=1, dim=2, device=dev)
        G = c.gates[2]
        c.gate(G.H, [0])
        c.gate(G.H, [0])

        for ket in [ket0(2), torch.tensor([0, 1], dtype=C64)]:

            result = c(ket)

            self.stateEqual(ket.numpy(), result, msg="H² ≠ I")

    def test_x_involution(self):
        """
        $X^2 = I$: applying $X$ twice returns the original state, two ways
        """
        c = Circuit(wires=1, dim=2, device=dev)
        G = c.gates[2]
        c.gate(G.X, [0])
        c.gate(G.X, [0])

        for ket in [ket0(2), torch.tensor([0, 1], dtype=C64)]:

            result = c(ket)

            self.stateEqual(ket.numpy(), result, msg="X² ≠ I")

    def test_circuit_same_as_operator(self):
        """
        Single-gate circuit $H$ on $|0\\rangle$ equals $G.H @ |0\\rangle$, two ways
        """
        G = Circuit(wires=1, dim=2, device=dev).gates[2]
        c = Circuit(wires=1, dim=2, device=dev)
        c.gate(G.H, [0])

        ket = ket0(2)
        via_circuit = c(ket)
        via_operator = G.H @ Ket2(0)

        self.stateEqual(via_operator, via_circuit, msg="Circuit vs operator mismatch")


class CircuitMeasurements(Question):
    """
    Circuit expectation value and sampling tests.
    """

    def test_expectation_z_on_plus(self):
        """
        $\\langle+|Z|+\\rangle = 0$: Z expectation on the $|+\\rangle$ state is zero
        """
        c = Circuit(wires=1, dim=2, device=dev)
        G = c.gates[2]
        c.gate(G.H, [0])
        Z = torch.tensor([[1, 0], [0, -1]], dtype=C64)

        result = c.expectation(Z, ket0(2))

        self.assertAlmostEqual(float(result.item()), 0.0, places=5, msg="<+|Z|+> should be 0")

    def test_expectation_x_on_plus(self):
        """
        $\\langle+|X|+\\rangle = 1$: X expectation on the $|+\\rangle$ state is one
        """
        c = Circuit(wires=1, dim=2, device=dev)
        G = c.gates[2]
        c.gate(G.H, [0])
        X = torch.tensor([[0, 1], [1, 0]], dtype=C64)

        result = c.expectation(X, ket0(2))

        self.assertAlmostEqual(float(result.item()), 1.0, places=5, msg="<+|X|+> should be 1")

    def test_sample_keys_and_total(self):
        """
        Sampling $|+\\rangle$ 1000 times returns both $|0\\rangle$ and $|1\\rangle$ with total count 1000
        """
        torch.manual_seed(0)
        c = Circuit(wires=1, dim=2, device=dev)
        G = c.gates[2]
        c.gate(G.H, [0])

        counts = c.sample(ket0(2), shots=1000)

        self.assertEqual(sorted(counts.keys()), ["0", "1"], msg="Sampling |+> should produce both |0> and |1> keys")

        self.assertEqual(sum(counts.values()), 1000, msg="Sample total should equal shots=1000")

    def test_sample_deterministic_state(self):
        """
        Sampling $|0\\rangle$ (identity circuit) 100 times returns only $|0\\rangle$
        """
        c = Circuit(wires=1, dim=2, device=dev)

        counts = c.sample(ket0(2), shots=100)

        self.assertEqual(counts, {"0": 100}, msg="Deterministic |0> sample should yield only '0'")

    def test_sample_qutrit_keys(self):
        """
        Sampling a 2-qutrit circuit returns bitstrings over $\\{0,1,2\\}^2$
        """
        torch.manual_seed(0)
        c = Circuit(wires=2, dim=3, device=dev)
        G = c.gates[3]
        c.gate(G.H, [0])
        counts = c.sample(ket0(9), shots=300)
        for key in counts:

            self.assertEqual(len(key), 2, msg="Qutrit 2-wire sample keys should have length 2")

            self.assertTrue(all(ch in "012" for ch in key), msg="Qutrit sample keys should only contain digits 0, 1, 2")

        self.assertEqual(sum(counts.values()), 300, msg="Qutrit sample total should equal shots=300")


if __name__ == "__main__":
    runner = Exam(
        name="Qudit Vector Circuit Tests",
        desc="Validation of circuit forward pass in statevector mode",
        file="circuit_v.md",
    )
    runner.run(load(VectorCircuit))
    runner.run(load(CircuitMeasurements))
