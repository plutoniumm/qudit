import sys

sys.path.append("..")

from qudit import Circuit
import numpy as np
import unittest


class TestDraw(unittest.TestCase):
    def test_draw_matches_expected(self):
        solution = """
|0> [2] ┤─H──RY(angle=1.5708)─┤
|0> [2] ┤─X───────────────────┤
|0> [3] ┤─X─╭●────────────────┤
|0> [3] ┤───╰U────────────────┤
""".strip()

        c2 = Circuit(wires=4, dim=[2, 2, 3, 3], device="cpu")
        G2 = c2.gates[2]
        G3 = c2.gates[3]
        c2.gate(G2.H, [0])
        c2.gate(G2.RY, [0], angle=np.pi / 2)
        c2.gate(G2.X, [1])
        c2.gate(G3.X, [2])
        c2.gate(G3.CX, [2, 3])

        fig = c2.draw()
        self.assertEqual(fig, solution)


if __name__ == "__main__":
    unittest.main()
