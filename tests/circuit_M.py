from MDR import Exam, load, Question
import sys

sys.path.append("..")

from qudit import Circuit, Mode
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

        self.matEqual(rhoV, rhoM, msg="Single-qubit: VECTOR and MATRIX modes should agree")

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

        self.matEqual(rhoV, rhoM, msg="Single-qutrit: VECTOR and MATRIX modes should agree")

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

        self.matEqual(rhoV, rhoM, msg="Mixed-dims [2,2,3,3]: VECTOR and MATRIX modes should agree")

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

        self.assertAlmostEqual(tr, 1.0, places=5, msg="Unitary channel should preserve trace")

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

        self.assertAlmostEqual(purity, 1.0, places=5, msg="Unitary channel should preserve purity")


if __name__ == "__main__":
    runner = Exam(
        name="Qudit Matrix Circuit Tests",
        desc="Validation of circuit forward pass in density-matrix mode",
        file="circuit_m.md",
    )
    runner.run(load(MatrixCircuit))
