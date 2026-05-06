from typing import List, Any, Sequence, Optional, Union
from .index import Channel, Multiplex
from itertools import product
from ..circuit.gates import Gate
from .kraus import GAD, Pauli, Depolarising, PhaseDamp, Reset, ThermalRelax
import torch as pt

C128 = pt.complex128


def permut(lst: List[str], n: int) -> List[List[str]]:
    """
    Return unique length-$n$ permutations of a list of symbols.

    Used to enumerate Kraus-operator "words" (tensor-product factor choices).
    """
    if n > len(lst):
        raise ValueError("n must be less than or equal to the length of lst")

    return [list(p) for p in product(set(lst), repeat=n)]


def mkron(args: Sequence[Any]) -> Any:
    """
    Compute a left-to-right Kronecker product $A_0 \otimes A_1 \otimes \cdots$.

    This is used to build many-body Kraus operators from single-site ones.
    """
    result = args[0]
    for i in range(1, len(args)):
        result = pt.kron(result, args[i])

    return result


def ungroup(lst: List[List[Any]]) -> List[Any]:
    return [item for sublist in lst for item in sublist]


class IID:
    @staticmethod
    def AD(n: int, d: int, y: float) -> "Multiplex":
        """
        Independent identical amplitude-damping on each of $n$ qudits of dimension $d$.

        Single-site Kraus set: $\\{A_k\\}_{k=0}^{d-1}$ from `GAD.A(k, d, y)`.
        For $d=2$ this recovers the standard qubit amplitude-damping channel.
        """
        channel_list = []
        kraus_mats = [GAD.A(k, d, y) for k in range(d)]
        for i in range(n):
            kraus_gates = [
                Gate(
                    m,
                    index=[i],
                    wires=n,
                    dim=d,
                    name=f"A{k}_{i}",
                    params=[("y", y), ("i", i)],
                )
                for k, m in enumerate(kraus_mats)
            ]
            channel_list.append(Channel([[g] for g in kraus_gates]))

        return Multiplex(channel_list)

    @staticmethod
    def GAD(n: int, d: int, y: float, p: float) -> "Multiplex":
        """
        Independent identical generalized amplitude-damping on each of $n$ qudits of dimension $d$.

        Single-site Kraus set: $\\{A_k, R_k\\}_{k=0}^{d-1}$ from `GAD.A/R(k, d, y, p)`.
        For $d=2$ this recovers the standard qubit GAD channel.
        """
        channel_list = []
        A_mats = [GAD.A(k, d, y, p) for k in range(d)]
        R_mats = [GAD.R(k, d, y, p) for k in range(d)]
        for i in range(n):
            kraus_gates = [
                Gate(
                    m,
                    index=[i],
                    wires=n,
                    dim=d,
                    name=f"A{k}_{i}",
                    params=[("y", y), ("p", p), ("i", i)],
                )
                for k, m in enumerate(A_mats)
            ] + [
                Gate(
                    m,
                    index=[i],
                    wires=n,
                    dim=d,
                    name=f"R{k}_{i}",
                    params=[("y", y), ("p", p), ("i", i)],
                )
                for k, m in enumerate(R_mats)
            ]
            channel_list.append(Channel([[g] for g in kraus_gates]))

        return Multiplex(channel_list)

    @staticmethod
    def Depolarising(n: int, d: int, p: float) -> "Multiplex":
        """
        Independent identical depolarising noise on each of $n$ qudits of dimension $d$.

        Single-site Kraus set: all $d^2$ Weyl-Heisenberg operators from `Depolarising.ops(d, p)`.
        """
        channel_list = []
        kraus_mats = Depolarising.ops(d, p)
        for i in range(n):
            kraus_gates = [
                Gate(
                    m,
                    index=[i],
                    wires=n,
                    dim=d,
                    name=f"W{k}_{i}",
                    params=[("p", p), ("i", i)],
                )
                for k, m in enumerate(kraus_mats)
            ]
            channel_list.append(Channel([[g] for g in kraus_gates]))

        return Multiplex(channel_list)

    @staticmethod
    def PhaseDamp(n: int, d: int, p: float) -> "Multiplex":
        """
        Independent identical phase damping on each of $n$ qudits of dimension $d$.

        Single-site Kraus set: $d$ operators from `PhaseDamp.ops(d, p)`.
        """
        channel_list = []
        kraus_mats = PhaseDamp.ops(d, p)
        for i in range(n):
            kraus_gates = [
                Gate(
                    m,
                    index=[i],
                    wires=n,
                    dim=d,
                    name=f"PD{k}_{i}",
                    params=[("p", p), ("i", i)],
                )
                for k, m in enumerate(kraus_mats)
            ]
            channel_list.append(Channel([[g] for g in kraus_gates]))

        return Multiplex(channel_list)

    @staticmethod
    def BitFlip(n: int, d: int, p: float) -> "Multiplex":
        """
        Independent identical bit-flip (shift) noise on each of $n$ qudits of dimension $d$.

        Single-site Kraus set: $\{\sqrt{1-p}\,I,\, \sqrt{p}\,X_d\}$ where $X_d$ is the
        cyclic shift operator. For $d=2$ this is the standard qubit bit-flip channel.
        """
        channel_list = []
        shift = pt.roll(pt.eye(d, dtype=C128), shifts=-1, dims=1)
        kraus_mats = [(1 - p) ** 0.5 * pt.eye(d, dtype=C128), p**0.5 * shift]
        for i in range(n):
            kraus_gates = [
                Gate(
                    m,
                    index=[i],
                    wires=n,
                    dim=d,
                    name=f"BF{k}_{i}",
                    params=[("p", p), ("i", i)],
                )
                for k, m in enumerate(kraus_mats)
            ]
            channel_list.append(Channel([[g] for g in kraus_gates]))

        return Multiplex(channel_list)

    @staticmethod
    def PhaseFlip(n: int, d: int, p: float) -> "Multiplex":
        """
        Independent identical phase-flip (clock) noise on each of $n$ qudits of dimension $d$.

        Single-site Kraus set: $\{\sqrt{1-p}\,I,\, \sqrt{p}\,Z_d\}$ where $Z_d$ is the
        clock (phase) operator. For $d=2$ this is the standard qubit phase-flip channel.
        """
        import cmath as cm

        w = cm.exp(2j * cm.pi / d)
        channel_list = []
        clock = pt.diag(pt.tensor([w**k for k in range(d)], dtype=C128))
        kraus_mats = [(1 - p) ** 0.5 * pt.eye(d, dtype=C128), p**0.5 * clock]
        for i in range(n):
            kraus_gates = [
                Gate(
                    m,
                    index=[i],
                    wires=n,
                    dim=d,
                    name=f"PF{k}_{i}",
                    params=[("p", p), ("i", i)],
                )
                for k, m in enumerate(kraus_mats)
            ]
            channel_list.append(Channel([[g] for g in kraus_gates]))

        return Multiplex(channel_list)

    @staticmethod
    def Reset(n: int, d: int, p: float) -> "Multiplex":
        """
        Independent identical reset noise on each of $n$ qudits of dimension $d$.

        Single-site Kraus set: $d+1$ operators from `Reset.ops(d, p)`.
        """
        channel_list = []
        kraus_mats = Reset.ops(d, p)
        for i in range(n):
            kraus_gates = [
                Gate(
                    m,
                    index=[i],
                    wires=n,
                    dim=d,
                    name=f"RS{k}_{i}",
                    params=[("p", p), ("i", i)],
                )
                for k, m in enumerate(kraus_mats)
            ]
            channel_list.append(Channel([[g] for g in kraus_gates]))

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
    ) -> Union[Channel, "Multiplex"]:
        """
        Build an $n$-site generalized amplitude damping channel.

        Constructs tensor-product Kraus operators from single-site $A_k$ (lowering) and
        $R_k$ (raising) terms, then groups/filters "correctable" subsets by total order.
        """
        assert isinstance(p, float), "p must be a float"

        assert isinstance(Y, float), "Y must be a float"

        assert p <= 1 and Y <= 1, "p,Y must be in [0, 1]"

        if iid:
            return IID.GAD(n, d, Y, p)

        def _op_gen(error_word: Sequence[str]) -> Optional[list[Gate]]:
            gates = []
            for i, tag in enumerate(error_word):
                ord_val = int(tag[-1])
                m = GAD.A(ord_val, d, Y, p) if "a" in tag else GAD.R(ord_val, d, Y, p)

                if pt.allclose(m, pt.zeros_like(m), atol=1e-8):
                    return None

                if not pt.allclose(m, pt.eye(d, dtype=m.dtype), atol=1e-8):
                    gates.append(Gate(m, index=[i], wires=n, dim=d, name=f"{tag}_{i}"))

            return gates

        base_tags = [f"a{k}" for k in range(d)] + [f"r{k}" for k in range(d)]
        keys = list(permut(base_tags * n, n))

        Ak = []
        valid_keys = []
        for key in keys:
            word_gates = _op_gen(key)
            if word_gates is not None:
                Ak.append(word_gates)
                valid_keys.append(key)

        Ek: list[list[int]] = [[] for _ in range((order + 1) * 2 - 1)]
        for idx, key in enumerate(valid_keys):
            s = sum(int(Em[-1]) for Em in key)
            if s <= order:
                if any("r" in i and int(i[-1]) > 0 for i in key):
                    Ek[2 * s - 1].append(idx)
                else:
                    Ek[2 * s - 0].append(idx)

        op_ch = Channel(Ak)
        op_ch.correctables = Ek if group else ungroup(Ek)

        return op_ch

    @staticmethod
    def AD(
        d: int, n: int, Y: float, order: int = 1, group: bool = False, iid: bool = False
    ) -> Union[Channel, "Multiplex"]:
        """
        Build an $n$-site (pure) amplitude damping channel.

        This is the $p=0$ special case of GAD using only lowering operators $A_k$.
        """
        assert isinstance(Y, float), "Y must be a float"

        assert Y <= 1, "Y must be in [0, 1]"

        if iid:
            return IID.AD(n, d, Y)

        def _op_gen(error_word: Sequence[str]) -> Optional[list[Gate]]:
            gates = []
            for i, tag in enumerate(error_word):
                m = GAD.A(int(tag[-1]), d, Y)
                if pt.allclose(m, pt.zeros_like(m), atol=1e-8):
                    return None

                if not pt.allclose(m, pt.eye(d, dtype=m.dtype), atol=1e-8):
                    gates.append(Gate(m, index=[i], wires=n, dim=d, name=f"{tag}_{i}"))

            return gates

        keys = list(permut([f"a{k}" for k in range(d)] * n, n))

        Ak = []
        valid_keys = []
        for key in keys:
            word_gates = _op_gen(key)
            if word_gates is not None:
                Ak.append(word_gates)
                valid_keys.append(key)

        Ek: list[list[int]] = [[] for _ in range(order + 1)]
        for idx, key in enumerate(valid_keys):
            s = sum(int(Em[-1]) for Em in key)
            if s <= order:
                Ek[s].append(idx)

        op_ch = Channel(Ak)
        op_ch.correctables = Ek if group else ungroup(Ek)

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
        derived from $p=[p_X,p_Y,p_Z]$, and groups "correctable" subsets by Hamming weight.
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
        weight = {
            "I": 0,
            "X": 1,
            "Y": 1,
            "Z": 1,
        }

        def _op_gen(word: Sequence[str]) -> Optional[list[Gate]]:
            gates = []
            for i, gate_str in enumerate(word):
                m = funcs[gate_str](p)
                if pt.allclose(m, pt.zeros_like(m), atol=1e-8):
                    return None

                if not pt.allclose(m, pt.eye(2, dtype=m.dtype), atol=1e-8):
                    gates.append(
                        Gate(m, index=[i], wires=n, dim=2, name=f"{gate_str}_{i}")
                    )

            return gates

        keys = list(permut((["I"] + paulis) * n, n))

        Ak = []
        valid_keys = []
        for key in keys:
            word_gates = _op_gen(key)
            if word_gates is not None:
                Ak.append(word_gates)
                valid_keys.append(key)

        Ek: list[list[int]] = [[] for _ in range(order + 1)]
        for idx, key in enumerate(valid_keys):
            s = sum(weight[i] for i in key)
            if s <= order:
                Ek[s].append(idx)

        op_ch = Channel(Ak)
        op_ch.correctables = Ek if group else ungroup(Ek)

        return op_ch

    @staticmethod
    def Depolarising(
        d: int,
        n: int,
        p: float,
        iid: bool = False,
    ) -> Union[Channel, "Multiplex"]:
        """
        Build an $n$-site depolarising channel.

        Applies all $d^2$ Weyl-Heisenberg operators as tensor-product Kraus operators.
        The channel is $(1-p)\\rho + \\frac{p}{d^2}\\sum_{jk} W_{jk}\\rho W_{jk}^\\dagger$.
        Works for any $d \geq 2$.
        """
        assert 0 <= p <= 1, "p must be in [0, 1]"

        if iid:
            return IID.Depolarising(n, d, p)

        kraus_single = Depolarising.ops(d, p)

        def _op_gen(word: Sequence[int]) -> Optional[list[Gate]]:
            gates = []
            for i, ki in enumerate(word):
                m = kraus_single[ki]
                if pt.allclose(m, pt.zeros_like(m), atol=1e-8):
                    return None

                if not pt.allclose(m, pt.eye(d, dtype=m.dtype), atol=1e-8):
                    gates.append(Gate(m, index=[i], wires=n, dim=d, name=f"W{ki}_{i}"))

            return gates

        num_ops = d * d
        keys = list(product(range(num_ops), repeat=n))

        Ak = []
        for key in keys:
            word_gates = _op_gen(key)
            if word_gates is not None:
                Ak.append(word_gates)

        return Channel(Ak)

    @staticmethod
    def PhaseDamp(
        d: int,
        n: int,
        p: float,
        iid: bool = False,
    ) -> Union[Channel, "Multiplex"]:
        """
        Build an $n$-site phase damping (dephasing) channel.

        Eliminates off-diagonal coherences without energy exchange.
        Works for any $d \geq 2$.
        """
        assert 0 <= p <= 1, "p must be in [0, 1]"

        if iid:
            return IID.PhaseDamp(n, d, p)

        kraus_single = PhaseDamp.ops(d, p)

        def _op_gen(word: Sequence[int]) -> Optional[list[Gate]]:
            gates = []
            for i, ki in enumerate(word):
                m = kraus_single[ki]
                if pt.allclose(m, pt.zeros_like(m), atol=1e-8):
                    return None

                if not pt.allclose(m, pt.eye(d, dtype=m.dtype), atol=1e-8):
                    gates.append(Gate(m, index=[i], wires=n, dim=d, name=f"PD{ki}_{i}"))

            return gates

        num_ops = d
        keys = list(product(range(num_ops), repeat=n))

        Ak = []
        for key in keys:
            word_gates = _op_gen(key)
            if word_gates is not None:
                Ak.append(word_gates)

        return Channel(Ak)

    @staticmethod
    def BitFlip(
        d: int,
        n: int,
        p: float,
        iid: bool = False,
    ) -> Union[Channel, "Multiplex"]:
        """
        Build an $n$-site bit-flip (shift) channel.

        Applies the cyclic shift $X_d$ with probability $p$. For $d=2$ this is the
        standard qubit bit-flip channel. Works for any $d \geq 2$.
        """
        assert 0 <= p <= 1, "p must be in [0, 1]"

        if iid:
            return IID.BitFlip(n, d, p)

        shift = pt.roll(pt.eye(d, dtype=C128), shifts=-1, dims=1)
        kraus_single = [(1 - p) ** 0.5 * pt.eye(d, dtype=C128), p**0.5 * shift]

        def _op_gen(word: Sequence[int]) -> Optional[list[Gate]]:
            gates = []
            for i, ki in enumerate(word):
                m = kraus_single[ki]
                if pt.allclose(m, pt.zeros_like(m), atol=1e-8):
                    return None

                if not pt.allclose(m, pt.eye(d, dtype=m.dtype), atol=1e-8):
                    gates.append(Gate(m, index=[i], wires=n, dim=d, name=f"BF{ki}_{i}"))

            return gates

        keys = list(product(range(2), repeat=n))

        Ak = []
        for key in keys:
            word_gates = _op_gen(key)
            if word_gates is not None:
                Ak.append(word_gates)

        return Channel(Ak)

    @staticmethod
    def PhaseFlip(
        d: int,
        n: int,
        p: float,
        iid: bool = False,
    ) -> Union[Channel, "Multiplex"]:
        """
        Build an $n$-site phase-flip (clock) channel.

        Applies the clock operator $Z_d$ with probability $p$. For $d=2$ this is the
        standard qubit phase-flip channel. Works for any $d \geq 2$.
        """
        assert 0 <= p <= 1, "p must be in [0, 1]"

        if iid:
            return IID.PhaseFlip(n, d, p)

        import cmath as cm

        w = cm.exp(2j * cm.pi / d)
        clock = pt.diag(pt.tensor([w**k for k in range(d)], dtype=C128))
        kraus_single = [(1 - p) ** 0.5 * pt.eye(d, dtype=C128), p**0.5 * clock]

        def _op_gen(word: Sequence[int]) -> Optional[list[Gate]]:
            gates = []
            for i, ki in enumerate(word):
                m = kraus_single[ki]
                if pt.allclose(m, pt.zeros_like(m), atol=1e-8):
                    return None

                if not pt.allclose(m, pt.eye(d, dtype=m.dtype), atol=1e-8):
                    gates.append(Gate(m, index=[i], wires=n, dim=d, name=f"PF{ki}_{i}"))

            return gates

        keys = list(product(range(2), repeat=n))

        Ak = []
        for key in keys:
            word_gates = _op_gen(key)
            if word_gates is not None:
                Ak.append(word_gates)

        return Channel(Ak)

    @staticmethod
    def Reset(
        d: int,
        n: int,
        p: float,
        iid: bool = False,
    ) -> Union[Channel, "Multiplex"]:
        """
        Build an $n$-site reset channel.

        Collapses each qudit to $|0\\rangle$ with probability $p$. Works for any $d \\geq 2$.
        """
        assert 0 <= p <= 1, "p must be in [0, 1]"

        if iid:
            return IID.Reset(n, d, p)

        kraus_single = Reset.ops(d, p)

        def _op_gen(word: Sequence[int]) -> Optional[list[Gate]]:
            gates = []
            for i, ki in enumerate(word):
                m = kraus_single[ki]
                if pt.allclose(m, pt.zeros_like(m), atol=1e-8):
                    return None

                if not pt.allclose(m, pt.eye(d, dtype=m.dtype), atol=1e-8):
                    gates.append(Gate(m, index=[i], wires=n, dim=d, name=f"RS{ki}_{i}"))

            return gates

        num_ops = d + 1
        keys = list(product(range(num_ops), repeat=n))

        Ak = []
        for key in keys:
            word_gates = _op_gen(key)
            if word_gates is not None:
                Ak.append(word_gates)

        return Channel(Ak)

    @staticmethod
    def ThermalRelax(
        n: int,
        T1: float,
        T2: float,
        t: float,
    ) -> Channel:
        """
        Build an $n$-site thermal relaxation channel ($d=2$ qubits only).

        Combines $T_1$ energy decay and $T_2$ dephasing. Raises `ValueError` for $d > 2$.
        Requires $T_2 \leq 2 T_1$.

        Args:
            n: number of qubits
            T1: longitudinal relaxation time
            T2: transverse dephasing time ($T_2 \leq 2T_1$ required)
            t: gate time
        """
        d = 2

        if T2 > 2 * T1:
            raise ValueError("ThermalRelax requires T2 <= 2*T1")

        kraus_single = ThermalRelax.ops(T1, T2, t)

        def _op_gen(word: Sequence[int]) -> Optional[list[Gate]]:
            gates = []
            for i, ki in enumerate(word):
                m = kraus_single[ki]
                if pt.allclose(m, pt.zeros_like(m), atol=1e-8):
                    return None

                if not pt.allclose(m, pt.eye(d, dtype=m.dtype), atol=1e-8):
                    gates.append(Gate(m, index=[i], wires=n, dim=d, name=f"TR{ki}_{i}"))

            return gates

        num_ops = len(kraus_single)
        keys = list(product(range(num_ops), repeat=n))

        Ak = []
        for key in keys:
            word_gates = _op_gen(key)
            if word_gates is not None:
                Ak.append(word_gates)

        return Channel(Ak)
