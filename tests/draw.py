from MDR import Exam, load, Question
import sys

sys.path.append("..")

from qudit import Circuit
import numpy as np


class CircuitDraw(Question):
    """
    Circuit visualization tests: ASCII diagram rendering via $\\mathrm{draw()}$.
    """

    def test_draw_matches_expected(self):
        """
        $\\mathrm{draw()}$ produces the expected ASCII circuit diagram for a
        mixed-dimension $[2,2,3,3]$ circuit with $H, R_Y, X, CX$ gates
        """
        solution = """
|0> [2] ┤─H──RY(angle=1.5708)─┤
|0> [2] ┤─X───────────────────┤
|0> [3] ┤─X─╭●────────────────┤
|0> [3] ┤───╰U────────────────┤
""".strip()

        c = Circuit(wires=4, dim=[2, 2, 3, 3], device="cpu")
        G2 = c.gates[2]
        G3 = c.gates[3]
        c.gate(G2.H, [0])
        c.gate(G2.RY, [0], angle=np.pi / 2)
        c.gate(G2.X, [1])
        c.gate(G3.X, [2])
        c.gate(G3.CX, [2, 3])

        self.assertEqual(c.draw(), solution, msg="Circuit draw() output does not match expected ASCII diagram")


if __name__ == "__main__":
    runner = Exam(
        name="Qudit Draw Tests",
        desc="Validation of ASCII circuit diagram rendering",
        file="draw.md",
    )
    runner.run(load(CircuitDraw))
