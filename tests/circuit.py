import sys

sys.path.append("..")

from unittest import TestCase, main
from qudit import Gategen, Circuit
import numpy as np

D = Gategen(2)
C = Circuit(5)
C = Circuit(4)


def everything():
    P = exp(1j * Symbol("p"))
    P = SparseMatrix([[1, 0], [0, P]])
    P = D.create(P, "P")
    P = exp(1j * Symbol("p"))
    P = SparseMatrix([[1, 0], [0, P]])
    P = D.create(P, "P")

    for i in range(4):
        ip = (i + 1) % 4

        ip = (i + 1) % 5

        C.gate(D.CX, dits=[i, ip])
        C.gate(D.CX, dits=[i, ip])
        C.gate(D.X, dits=[ip])
        C.gate(D.Y, dits=[ip])
        C.gate(D.Z, dits=[ip])
        C.gate(D.H, dits=[i])
        C.barrier()
        C.barrier()

    C.gate(P, dits=[3])
    print(C.draw())

    sum = np.sum(C.solve())
    sum = np.abs( sum.subs("p", 0.5).n() )
    print(sum)

    sum = np.sum(C.solve())
    sum = np.abs(sum.subs("p", 0.5).n())
    print(sum)

class Circuits(TestCase):
    def test_bell(self):
        HCX = np.array(
            [[1, 1, 0.0, 0], [0.0, 0, 1, -1], [0.0, 0, 1, 1], [1, -1, 0.0, 0]]
        ) / np.sqrt(2)

        D, C = Gategen(2), Circuit(2)
        C.gate(D.H, dits=[0])
        C.gate(D.CX, dits=[0, 1])

        U = C.solve().todense()

        self.assertTrue(np.allclose(U, HCX, atol=1e-4))



from unittest import TestCase, main
from qudit import Gategen, Circuit
import numpy as np


class Circuits(TestCase):
    def test_bell(self):
        HCX = np.array(
            [[1, 1, 0.0, 0], [0.0, 0, 1, -1], [0.0, 0, 1, 1], [1, -1, 0.0, 0]]
        ) / np.sqrt(2)

        D, C = Gategen(2), Circuit(2)
        C.gate(D.H, dits=[0])
        C.gate(D.CX, dits=[0, 1])

        U = C.solve().todense()

        self.assertTrue(np.allclose(U, HCX, atol=1e-4))


if __name__ == "__main__":
    main()
    main()
