import sys

sys.path.append("..")


# from sympy import exp, SparseMatrix, Symbol
from unittest import TestCase, main
from qudit import Circuit
import numpy as np

C = Circuit(4, dim=2)
D = C.gates


def everything():
    # P = exp(1j * Symbol("p"))
    # P = SparseMatrix([[1, 0], [0, P]])
    # P = D.create(P, "P")

    for i in range(4):
        ip = (i + 1) % 4

        C.gate(D.CX, dits=[i, ip])
        C.gate(D.X, dits=[ip])
        C.gate(D.Y, dits=[i])
        C.gate(D.Z, dits=[ip])
        C.gate(D.H, dits=[i])
        C.barrier()

    C.gate(D.CZ, dits=[1, 3])
    # C.gate(P, dits=[3])
    print(C.draw())
    print(np.sum(C.solve()))

    # sum = np.sum(C.solve())
    # sum = np.abs(sum.subs("p", 0.5).n())


everything()


class Circuits(TestCase):
    def test_bell(self):
        HCX = np.array(
            [[1, 1, 0, 0], [0, 0, 1, -1], [0, 0, 1, 1], [1, -1, 0, 0]]
        ) / np.sqrt(2)

        C = Circuit(2, dim=2)
        G = C.gates
        C.gate(G.H, dits=[0])
        C.gate(G.CX, dits=[0, 1])

        U = C.solve().todense()
        self.assertTrue(np.allclose(U, HCX, atol=1e-4))


# if __name__ == "__main__":
#     main()
