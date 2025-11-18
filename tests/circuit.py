import sys

sys.path.append("..")

from unittest import TestCase, main
from qudit import Circuit
import numpy as np
import torch

dev = "cpu"
C64 = torch.complex64

def nonZero(data, tol: float = 1e-5, round: int = 3, name: str = ""):
    indices = torch.where(abs(data) > tol)[0]
    data = np.round(data.cpu().numpy().flatten(), round)

    if name != "":
        print(f"{name}:")
    for idx in indices:
        print(f" |Ψ⟩[{idx}]: {data[idx]:.3f}")

def ket0(size):
    x = torch.zeros(size, dtype=C64, device=dev)
    x[0] = 1.0
    return x


class TestCircuit(TestCase):
    def assert_state(self, data, expected, tol: float = 1e-3):
        if isinstance(data, torch.Tensor):
            data = data.flatten().detach().cpu().numpy()
        nz = set(np.where(np.abs(data) > tol)[0].tolist())
        exp_idx = set(expected.keys())
        self.assertSetEqual(nz, exp_idx)

        for idx, val in expected.items():
            self.assertAlmostEqual(float(np.real(data[idx])), float(np.real(val)), delta=tol)
            self.assertAlmostEqual(float(np.imag(data[idx])), float(np.imag(val)), delta=tol)

        self.assertAlmostEqual(float(np.vdot(data, data).real), 1.0, delta=1e-3)

    def test_DOSE(self):
        c1 = Circuit(wires=2, dim=2, device=dev)
        G2 = c1.gates[2]
        c1.gate(G2.H, [0])
        c1.gate(G2.CX, [0, 1])
        x1 = ket0(c1.width)

        print(c1.matrix())
        raise SystemExit
        psi = c1(x1)
        a = 1 / np.sqrt(2)
        self.assert_state(psi, {0: a + 0j, 3: a + 0j})

    def test_bell_state(self):
        return None
        c1 = Circuit(wires=2, dim=2, device=dev)
        G2 = c1.gates[2]
        c1.gate(G2.H, [0])
        c1.gate(G2.CX, [0, 1])
        x1 = ket0(c1.width)

        psi = c1(x1)
        a = 1 / np.sqrt(2)
        self.assert_state(psi, {0: a + 0j, 3: a + 0j})

    def test_mixed_dimension_ent(self):
        return None
        c2 = Circuit(wires=4, dim=[2, 2, 3, 3], device=dev)
        G2 = c2.gates[2]
        G3 = c2.gates[3]
        c2.gate(G2.H, [0])
        c2.gate(G2.X, [1])
        c2.gate(G3.CX, [2, 3])
        x2 = ket0(c2.width)
        psi = c2(x2)
        a = 1 / np.sqrt(2)
        self.assert_state(psi, {9: a + 0j, 27: a + 0j})

    def test_three_qutrit_ghz(self):
        return None
        c3 = Circuit(wires=3, dim=3, device=dev)
        G3 = c3.gates[3]
        c3.gate(G3.H, [0])
        c3.gate(G3.CX, [0, 1])
        c3.gate(G3.CX, [0, 2])
        x3 = ket0(c3.width)
        psi = c3(x3)
        a = 1 / np.sqrt(3)
        self.assert_state(psi, {0: a + 0j, 17: a + 0j, 18: a + 0j})


if __name__ == "__main__":
    main()
