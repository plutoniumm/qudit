from MDR import Exam, load, Question
import sys

sys.path.append("..")

from qudit import Gategen, Basis
import numpy as np
import torch

C64 = torch.complex64

class D2_Gates(Question):
    """
    Single-qubit gate correctness tests for $d=2$.
    """
    Ket = Basis(2)
    G = Gategen(2)

    def test_X__0_1(self):
        """
        $X\\vert0\\rangle = \\vert1\\rangle$
        """
        P0, P1 = self.Ket(0), self.Ket(1)
        _P1 = self.G.X @ P0

        self.stateEqual(P1, _P1,
        msg=f"X gate mismatch. got P0->P1={_P1} exp P1={P1}")

    def test_X__1_0(self):
        """
        $X\\vert1\\rangle = \\vert0\\rangle$
        """
        P0, P1 = self.Ket(0), self.Ket(1)
        _P0 = self.G.X @ P1
        self.stateEqual(P0, _P0,
            msg=f"X gate mismatch. got P1->P0={_P0} exp P0={P0}",
        )

    def test_Y__0_i1(self):
        """
        $Y\\vert0\\rangle = i\\vert1\\rangle$
        """
        P0, P1 = self.Ket(0), self.Ket(1)
        _P1 = self.G.Y @ P0
        self.stateEqual(1j * P1, _P1,
            msg=f"Y gate mismatch. got Y0={_P1}",
        )

    def test_Y__1_mi0(self):
        """
        $Y\\vert1\\rangle = -i\\vert0\\rangle$
        """
        P0, P1 = self.Ket(0), self.Ket(1)
        _P0 = self.G.Y @ P1
        self.stateEqual(-1j * P0, _P0,
            msg=f"Y gate mismatch. got Y1={_P0}",
        )

    def test_Z__0_stays_0(self):
        """
        $Z\\vert0\\rangle = \\vert0\\rangle$
        """
        P0 = self.Ket(0)
        _P0 = self.G.Z @ P0
        self.stateEqual(P0, _P0,
            msg=f"Z gate mismatch. got P0={_P0} exp P0={P0}",
        )

    def test_Z__1_m1(self):
        """
        $Z\\vert1\\rangle = -\\vert1\\rangle$
        """
        P1 = self.Ket(1)
        _P1 = self.G.Z @ P1
        self.stateEqual(-P1, _P1,
            msg=f"Z gate mismatch. got P1={_P1} exp P1={-P1}",
        )

    def test_H__0_sup(self):
        """
        $H\\vert0\\rangle = (\\vert0\\rangle + \\vert1\\rangle)/\sqrt{2}$
        """
        P0, P1 = self.Ket(0), self.Ket(1)
        a = 1 / np.sqrt(2)
        exp0 = a * (P0 + P1)
        _P0 = self.G.H @ P0
        self.stateEqual(exp0, _P0,
            msg=f"H gate mismatch. got H0={_P0} exp H0={exp0}",
        )

    def test_H__1_sup(self):
        """
        $H\\vert1\\rangle = (\\vert0\\rangle - \\vert1\\rangle)/\sqrt{2}$
        """
        P0, P1 = self.Ket(0), self.Ket(1)
        a = 1 / np.sqrt(2)
        exp1 = a * (P0 - P1)
        _P1 = self.G.H @ P1
        self.stateEqual(exp1, _P1,
            msg=f"H gate mismatch. got H1={_P1} exp H1={exp1}",
        )

    def test_I__0_ID(self):
        """
        $I\\vert0\\rangle = \\vert0\\rangle$
        """
        P0 = self.Ket(0)
        _P0 = self.G.I @ P0
        self.stateEqual(P0, _P0,
            msg=f"I gate mismatch.",
        )

    def test_I__1_ID(self):
        """
        $I\\vert1\\rangle = \\vert1\\rangle$
        """
        P1 = self.Ket(1)
        _P1 = self.G.I @ P1
        self.stateEqual(P1, _P1,
            msg=f"I gate mismatch.",
        )

    def test_CX__00(self):
        """
        $CX\\vert00\\rangle = \\vert00\\rangle$
        """
        P00 = self.Ket(0, 0)
        _P00 = self.G.CX @ P00
        self.stateEqual(P00, _P00,
            msg="CX gate mismatch."
        )

    def test_CX__01(self):
        """
        $CX\\vert01\\rangle = \\vert01\\rangle$
        """
        P01 = self.Ket(0, 1)
        _P01 = self.G.CX @ P01
        self.stateEqual(P01, _P01,
            msg="CX gate mismatch."
        )

    def test_CX__10_11(self):
        """
        $CX\\vert10\\rangle = \\vert11\\rangle$
        """
        P10 = self.Ket(1, 0)
        _P11 = self.G.CX @ P10
        self.stateEqual(self.Ket(1, 1), _P11,
            msg="CX gate mismatch."
        )

    def test_CX__11_10(self):
        """
        $CX\\vert11\\rangle = \\vert10\\rangle$
        """
        P11 = self.Ket(1, 1)
        _P10 = self.G.CX @ P11
        self.stateEqual(self.Ket(1, 0), _P10,
            msg="CX gate mismatch."
        )

    def test_SWAP__01_10(self):
        """
        $SWAP\\vert01\\rangle = \\vert10\\rangle$
        """
        P01 = self.Ket(0, 1)
        _P10 = self.G.SWAP @ P01
        self.stateEqual(self.Ket(1, 0), _P10,
            msg="SWAP gate mismatch."
        )

    def test_SWAP__10_01(self):
        """
        $SWAP\\vert10\\rangle = \\vert01\\rangle$
        """
        P10 = self.Ket(1, 0)
        _P01 = self.G.SWAP @ P10
        self.stateEqual(self.Ket(0, 1), _P01,
            msg="SWAP gate mismatch."
        )

class D3_Gates(Question):
    """
    Single-qubit gate correctness tests for $d=3$.
    """
    Ket = Basis(3)
    G = Gategen(3)

    def test_I__0(self):
        """
        $I\\vert0\\rangle = \\vert0\\rangle$
        """
        P = self.Ket(0)
        _P = self.G.I @ P
        self.stateEqual(P, _P, msg=f"I mismatch for 0.")

    def test_I__1(self):
        """
        $I\\vert1\\rangle = \\vert1\\rangle$
        """
        P = self.Ket(1)
        _P = self.G.I @ P
        self.stateEqual(P, _P, msg=f"I mismatch for 1.")

    def test_I__2(self):
        """
        $I\\vert2\\rangle = \\vert2\\rangle$
        """
        P = self.Ket(2)
        _P = self.G.I @ P
        self.stateEqual(P, _P, msg=f"I mismatch for 2.")

    def test_X__0_1(self):
        """
        $X\\vert0\\rangle = \\vert1\\rangle$
        """
        P0, P1 = self.Ket(0), self.Ket(1)
        self.stateEqual(P1, self.G.X @ P0, msg="X0 != 1")

    def test_X__1_2(self):
        """
        $X\\vert1\\rangle = \\vert2\\rangle$
        """
        P1, P2 = self.Ket(1), self.Ket(2)
        self.stateEqual(P2, self.G.X @ P1, msg="X1 != 2")

    def test_X__2_0(self):
        """
        $X\\vert2\\rangle = \\vert0\\rangle$
        """
        P2, P0 = self.Ket(2), self.Ket(0)
        self.stateEqual(P0, self.G.X @ P2, msg="X2 != 0")

    def test_Z__0(self):
        """
        $Z\\vert0\\rangle = \\vert0\\rangle$
        """
        P0 = self.Ket(0)
        self.stateEqual(P0, self.G.Z @ P0, msg="Z0 != 0")

    def test_Z__1(self):
        """
        $Z\\vert1\\rangle = \omega\\vert1\\rangle$
        """
        P1 = self.Ket(1)
        w = np.exp(2j * np.pi / 3)
        self.stateEqual(w * P1, self.G.Z @ P1, msg="Z1 != w1")

    def test_Z__2(self):
        """
        $Z\\vert2\\rangle = \omega^2\\vert2\\rangle$
        """
        P2 = self.Ket(2)
        w = np.exp(2j * np.pi / 3)
        self.stateEqual(w**2 * P2, self.G.Z @ P2, msg="Z2 != w2")

    def test_Y__0_i1(self):
        """
        $Y\\vert0\\rangle = i\\vert1\\rangle$
        """
        P0, P1 = self.Ket(0), self.Ket(1)
        self.stateEqual(1j * P1, self.G.Y @ P0, msg="Y0 != i1")

    def test_Y__1_i2(self):
        """
        $Y\\vert1\\rangle = i\\vert2\\rangle$
        """
        P1, P2 = self.Ket(1), self.Ket(2)
        self.stateEqual(1j * P2, self.G.Y @ P1, msg="Y1 != i2")

    def test_Y__2_i0(self):
        """
        $Y\\vert2\\rangle = i\\vert0\\rangle$
        """
        P2, P0 = self.Ket(2), self.Ket(0)
        self.stateEqual(1j * P0, self.G.Y @ P2, msg="Y2 != i0")

    def test_H__0_sup(self):
        """
        $H\\vert0\\rangle = (\\vert0\\rangle + \\vert1\\rangle + \\vert2\\rangle)/\sqrt{3}$
        """
        exp0 = (1 / np.sqrt(3)) * (self.Ket(0) + self.Ket(1) + self.Ket(2))
        _P0 = self.G.H @ self.Ket(0)
        self.stateEqual(exp0, _P0, msg="H0 mismatch")

    def test_H__1_sup(self):
        """
        $H\\vert1\\rangle = (\\vert0\\rangle + \omega\\vert1\\rangle + \omega^2\\vert2\\rangle)/\sqrt{3}$
        """
        w = np.exp(2j * np.pi / 3)
        exp1 = (1 / np.sqrt(3)) * (self.Ket(0) + w * self.Ket(1) + w**2 * self.Ket(2))
        _P1 = self.G.H @ self.Ket(1)
        self.stateEqual(exp1, _P1, msg="H1 mismatch")

    def test_CX__00(self):
        """
        $CX\\vert00\\rangle = \\vert00\\rangle$
        """
        P00 = self.Ket(0, 0)
        _P00 = self.G.CX @ P00
        self.stateEqual(P00, _P00, msg="CX00 mismatch")

    def test_CX__11(self):
        """
        $CX\\vert11\\rangle = \\vert12\\rangle$
        """
        P11 = self.Ket(1, 1)
        P12 = self.Ket(1, 2)
        _P12 = self.G.CX @ P11
        self.stateEqual(P12, _P12, msg="CX11 mismatch")

    def test_CX__21(self):
        """
        $CX\\vert21\\rangle = \\vert20\\rangle$
        """
        P21 = self.Ket(2, 1)
        P20 = self.Ket(2, 0)
        _P20 = self.G.CX @ P21
        self.stateEqual(P20, _P20, msg="CX21 mismatch")

    def test_SWAP__12_21(self):
        """
        $SWAP\\vert12\\rangle = \\vert21\\rangle$
        """
        P12 = self.Ket(1, 2)
        P21 = self.Ket(2, 1)
        _P21 = self.G.SWAP @ P12
        self.stateEqual(P21, _P21, msg="SWAP12 mismatch")

if __name__ == "__main__":
    runner = Exam(
        name="Qudit Gate Tests",
        desc="Validation of single-qubit gate generators",
        file="gates.md",
    )
    runner.run(load(D2_Gates))
    runner.run(load(D3_Gates))
