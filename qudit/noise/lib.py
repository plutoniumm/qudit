from typing import List, Any, Sequence, Optional, Union
from .index import Channel, Error, Multiplex
from itertools import permutations
from .kraus import GAD, Pauli
import numpy as np

C128 = np.complex128


def permut(lst: List[str], n: int) -> List[List[str]]:
    """Return unique length-$n$ permutations of a list of symbols.

    Used to enumerate Kraus-operator “words” (tensor-product factor choices).
    """
    if n > len(lst):
        raise ValueError("n must be less than or equal to the length of lst")

    return [list(p) for p in set(permutations(lst, n))]


def mkron(args: Sequence[Any]) -> Any:
    """
    Compute a left-to-right Kronecker product $A_0 \otimes A_1 \otimes \cdots$.

    This is used to build many-body Kraus operators from single-site ones.
    """
    result = args[0]
    for i in range(1, len(args)):
        result = np.kron(result, args[i])

    return result


def ungroup(lst: List[List[Any]]) -> List[Any]:
    return [item for sublist in lst for item in sublist]


class IID:
    @staticmethod
    def AD(n: int, y: float) -> Multiplex:
        A0 = np.array([[1, 0], [0, np.sqrt(1 - y)]])
        A1 = np.array([[0, np.sqrt(y)], [0, 0]])
        I2 = np.eye(2)

        channel_list = []
        for i in range(n):
            A, B = (n - i - 1) * [I2], i * [I2]
            Is = "I" * n
            nA = Is[: n - i - 1] + "A_0" + Is[n - i :]
            nB = Is[: n - i - 1] + "A_1" + Is[n - i :]
            print(nA, nB)

            E0 = A + [A0] + B
            E1 = A + [A1] + B

            A = Error(2, mkron(E0), name=nA, params={"y": y, "i": i})
            B = Error(2, mkron(E1), name=nB, params={"y": y, "i": i})

            channel_list.append(Channel([A, B]))

        return Multiplex(channel_list)

    @staticmethod
    def GAD(n: int, y: float, p: float) -> Multiplex:
        A0 = np.sqrt(1 - p) * np.array([[1, 0], [0, np.sqrt(1 - y)]])
        A1 = np.sqrt(1 - p) * np.array([[0, np.sqrt(y)], [0, 0]])

        R0 = np.sqrt(p) * np.array([[np.sqrt(1 - y), 0], [0, 1]])
        R1 = np.sqrt(p) * np.array([[0, 0], [np.sqrt(y), 0]])

        I2 = np.eye(2)

        channel_list = []
        for i in range(n):
            A, B = (n - i - 1) * [I2], i * [I2]
            Is = "I" * n
            nA0 = Is[: n - i - 1] + "A_0" + Is[n - i :]
            nA1 = Is[: n - i - 1] + "A_1" + Is[n - i :]
            nR0 = Is[: n - i - 1] + "R_0" + Is[n - i :]
            nR1 = Is[: n - i - 1] + "R_1" + Is[n - i :]

            A0 = Error(
                2, mkron(A + [A0] + B), name=nA0, params={"y": y, "p": p, "i": i}
            )
            A1 = Error(
                2, mkron(A + [A1] + B), name=nA1, params={"y": y, "p": p, "i": i}
            )
            R0 = Error(
                2, mkron(A + [R0] + B), name=nR0, params={"y": y, "p": p, "i": i}
            )
            R1 = Error(
                2, mkron(A + [R1] + B), name=nR1, params={"y": y, "p": p, "i": i}
            )

            channel_list.append(Channel([A0, A1, R0, R1]))

        return Multiplex(channel_list)


class Process:
    """
    Factories for common multi-qudit noise processes.

    Each constructor returns a `Channel` in Kraus form $\Phi(\\rho)=\sum_k E_k\\rho E_k^\dagger$,
    and records which Kraus terms are considered correctable up to a given order.
    """

    @staticmethod
    def GAD(
        d: int,
        n: int,
        Y: float,
        p: float,
        order: int = 1,
        group: bool = False,
        iid: bool = False,
    ) -> Union[Channel, Multiplex]:
        """
        Build an $n$-site generalized amplitude damping channel.

        Constructs tensor-product Kraus operators from single-site $A_k$ (lowering) and
        $R_k$ (raising) terms, then groups/filters “correctable” subsets by total order.
        """
        assert isinstance(p, float), "p must be a float"
        assert isinstance(Y, float), "Y must be a float"
        assert p <= 1 and Y <= 1, "p,Y must be in [0, 1]"

        if iid:
            assert d == 2, "IID GAD is only implemented for qubits (d=2)"
            return IID.GAD(n, Y, p)

        def _op_gen(error_word: Sequence[str]) -> Any:
            temp: list[Any] = []
            for tag in error_word:
                order = int(tag[-1])
                if "a" in tag:
                    temp.append(GAD.A(order, d, Y, p))
                elif "r" in tag:
                    temp.append(GAD.R(order, d, Y, p))
                else:
                    raise ValueError(f"Unknown tag {tag} in error word {error_word}")
            return mkron(temp)

        keys = list(permut(["a0", "a1", "r0", "r1"] * n, n))
        Ak = [_op_gen(key) for key in keys]

        Ek: list[list[int]] = [[] for _ in range((order + 1) * 2 - 1)]
        for key in keys:
            s = np.sum([int(Em[-1]) for Em in key])
            if s <= order and not np.all(np.isclose(Ak[keys.index(key)], 0, atol=1e-8)):

                if any("r" in i and int(i[-1]) > 0 for i in key):
                    Ek[2 * s - 1].append(keys.index(key))
                else:
                    Ek[2 * s - 0].append(keys.index(key))

        op_ch = Channel(Ak)
        op_ch.correctables = Ek if group else ungroup(Ek)  # type: ignore[assignment]

        return op_ch

    @staticmethod
    def AD(
        d: int, n: int, Y: float, order: int = 1, group: bool = False, iid: bool = False
    ) -> Union[Channel, Multiplex]:
        """
        Build an $n$-site (pure) amplitude damping channel.

        This is the $p=0$ special case of GAD using only lowering operators $A_k$.
        """
        assert isinstance(Y, float), "Y must be a float"
        assert Y <= 1, "Y must be in [0, 1]"

        if iid:
            assert d == 2, "IID AD is only implemented for qubits (d=2)"
            return IID.AD(n, Y)

        def _op_gen(error_word: Sequence[str]) -> Any:
            """Map a word of 'a0','a1',... tags to a tensor-product Kraus operator."""
            individual = [GAD.A(int(tag[-1]), d, Y) for tag in error_word]
            return mkron(individual)

        keys = list(permut(["a0", "a1"] * n, n))
        Ak = [_op_gen(key) for key in keys]

        Ek: list[list[int]] = [[] for _ in range(order + 1)]
        for key in keys:
            s = np.sum([int(Em[-1]) for Em in key])
            if s <= order and not np.all(np.isclose(Ak[keys.index(key)], 0, atol=1e-8)):
                Ek[s].append(keys.index(key))

        op_ch = Channel(Ak)
        op_ch.correctables = Ek if group else ungroup(Ek)  # type: ignore[assignment]

        return op_ch

    @staticmethod
    def Pauli(
        n: int,
        paulis: Optional[list[str]] = None,
        p: Optional[list[float]] = None,
        order: int = 1,
        group: bool = False,
    ) -> Channel:
        """
        Build an $n$-site Pauli channel.

        Constructs tensor-product Kraus operators from $\{I,X,Y,Z\}$ with weights
        derived from $p=[p_X,p_Y,p_Z]$, and groups “correctable” subsets by Hamming weight.
        """
        if paulis is None:
            paulis = ["X", "Y", "Z"]
        if p is None:
            p = [0.0, 0.0, 0.0]

        funcs = {
            "I": Pauli.I,
            "X": lambda p: Pauli.X(p[0]),
            "Y": lambda p: Pauli.Y(p[1]),
            "Z": lambda p: Pauli.Z(p[2]),
        }
        weight = {"I": 0, "X": 1, "Y": 1, "Z": 1}

        def _op_gen(word: Sequence[str]) -> Any:
            """Map a Pauli word to the tensor-product Kraus operator."""
            return mkron([funcs[gate](p) for gate in word])

        keys = permut((["I"] + paulis) * n, n)
        Ak = [_op_gen(key) for key in keys]

        Ek: list[list[int]] = [[] for _ in range(order + 1)]
        for key in keys:
            s = np.sum([weight[i] for i in key])
            if s <= order and not np.all(np.isclose(Ak[keys.index(key)], 0, atol=1e-8)):
                Ek[s].append(keys.index(key))

        op_ch = Channel(Ak)
        op_ch.correctables = Ek if group else ungroup(Ek)  # type: ignore[assignment]

        return op_ch
