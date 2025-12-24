import sys

sys.path.append("..")

from unittest import TestCase, main
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


class TestCircuitMatrixMode(TestCase):
    def Close(self, a, b):
        self.assertTrue(torch.allclose(a, b, rtol=1e-4, atol=1e-4), "Tensors not close")

    def test_single_qubit(self):
        cM = Circuit(wires=1, dim=2, mode=Mode.MATRIX)
        cV = Circuit(wires=1, dim=2, mode=Mode.VECTOR)
        G2 = cM.gates[2]

        cM.gate(G2.H, [0])
        cV.gate(G2.H, [0])

        xV = ket0(cM.width)
        xM = toRho(xV)

        SV = cV(xV)
        Rho = cM(xM)

        SVR = toRho(SV)
        self.Close(SVR, Rho)

    def test_single_qutrit(self):
        cM = Circuit(wires=1, dim=3, mode=Mode.MATRIX)
        cV = Circuit(wires=1, dim=3, mode=Mode.VECTOR)
        G3 = cM.gates[3]

        cM.gate(G3.H, [0])
        cV.gate(G3.H, [0])

        xV = ket0(cM.width)
        xM = toRho(xV)

        SV, Rho = cV(xV), cM(xM)
        SVR = toRho(SV)

        self.Close(SVR, Rho)

    def test_mixed_dims(self):
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

        SV, Rho = cV(xV), cM(xM)
        SVR = toRho(SV)

        self.Close(SVR, Rho)


if __name__ == "__main__":
    main()
