from scipy.special._comb import _comb_int as nCr
from typing import List
import torch as pt

C128 = pt.complex128


class GAD:
    """
    Generalized amplitude damping (GAD) Kraus operators for a $d$-level system.

    Provides Kraus operators $\{A_k, R_k\}$ parameterized by environment excitation
    probability $p$ and damping parameter $Y$.
    """

    @staticmethod
    def A(order: int, d: int, Y: float, p: float = 0.0) -> pt.Tensor:
        """
        Construct the GAD lowering Kraus operator $A_k$.

        This operator maps population from $|r\\rangle$ to $|r-k\\rangle$ with weights
        determined by $(Y,p)$.
        """
        k = order

        assert isinstance(k, int) and k >= 0, "k must be int>=0"

        assert isinstance(d, int) and d > 0, "d must be int>0"

        obj = pt.zeros((d, d), dtype=C128)
        if k < d:
            rs = pt.arange(k, d, dtype=pt.float64)
            binom = pt.tensor(
                [float(nCr(int(r), k)) for r in range(k, d)], dtype=pt.float64
            )
            vals = binom.sqrt() * (1 - Y) ** ((rs - k) / 2) * Y ** (k / 2)
            obj[(rs - k).long(), rs.long()] = vals.to(C128)

        return (1 - p) ** 0.5 * obj

    @staticmethod
    def R(k: int, d: int, Y: float, p: float = 0.0) -> pt.Tensor:
        """
        Construct the GAD raising Kraus operator $R_k$.

        This operator maps population from $|r\\rangle$ to $|r+k\\rangle$ with weights
        determined by $(Y,p)$.
        """
        obj = pt.zeros((d, d), dtype=C128)
        if k < d:
            rs = pt.arange(0, d - k, dtype=pt.float64)
            binom = pt.tensor(
                [float(nCr(int(d - r - 1), k)) for r in range(d - k)], dtype=pt.float64
            )
            vals = binom * (1 - Y) ** ((d - rs - k - 1) / 2) * Y ** (k / 2)
            obj[(rs + k).long(), rs.long()] = vals.to(C128)

        return p**0.5 * obj


class Pauli:
    """
    Single-qubit Pauli error operators as Kraus operators.

    Each method returns $\sqrt{p}\,\sigma$ for $\sigma \in \{X,Y,Z\}$ and an identity
    term $\sqrt{1-\sum p_i}\,I$.
    """

    @staticmethod
    def X(p: float) -> pt.Tensor:
        """
        Return the bit-flip Kraus operator $\sqrt{p}\,X$.
        """
        return p**0.5 * pt.tensor([[0, 1], [1, 0]], dtype=C128)

    @staticmethod
    def Y(p: float) -> pt.Tensor:
        """
        Return the phase+bit-flip Kraus operator $\sqrt{p}\,Y$.
        """
        return p**0.5 * pt.tensor([[0, -1j], [1j, 0]], dtype=C128)

    @staticmethod
    def Z(p: float) -> pt.Tensor:
        """
        Return the phase-flip Kraus operator $\sqrt{p}\,Z$.
        """
        return p**0.5 * pt.tensor([[1, 0], [0, -1]], dtype=C128)

    @staticmethod
    def I(ps: List[float]) -> pt.Tensor:
        """
        Return the identity Kraus operator $\sqrt{1-\sum_i p_i}\,I$.
        """
        return (1 - sum(ps)) ** 0.5 * pt.tensor([[1, 0], [0, 1]], dtype=C128)


class Depolarising:
    """
    Depolarising channel Kraus operators for a $d$-level system.

    The channel acts as $\\Phi(\\rho) = (1-p)\\rho + \\frac{p}{d^2}\\sum_{j,k} W_{jk}\\rho W_{jk}^\\dagger$
    where $W_{jk}$ are the $d^2$ Weyl-Heisenberg operators.
    """

    @staticmethod
    def ops(d: int, p: float) -> List[pt.Tensor]:
        """
        Return all $d^2$ Weyl-Heisenberg Kraus operators weighted for the depolarising channel.

        The identity term carries weight $\sqrt{1 - p(d^2-1)/d^2}$ and each non-identity
        Weyl operator $W_{jk}$ carries weight $\sqrt{p/d^2}$.
        """
        import cmath

        w = cmath.exp(2j * cmath.pi / d)

        shift = pt.roll(pt.eye(d, dtype=C128), shifts=-1, dims=1)
        clock = pt.diag(pt.tensor([w**k for k in range(d)], dtype=C128))

        weyl = []
        for j in range(d):
            for k in range(d):
                W = pt.linalg.matrix_power(shift, j) @ pt.linalg.matrix_power(clock, k)
                weyl.append(W)

        result = []
        for i, W in enumerate(weyl):
            if i == 0:
                weight = (1 - p * (d * d - 1) / (d * d)) ** 0.5
            else:
                weight = (p / (d * d)) ** 0.5
            result.append(weight * W)

        return result


class PhaseDamp:
    """
    Phase damping (dephasing) Kraus operators for a $d$-level system.

    Kills off-diagonal coherences without energy exchange.
    For $d=2$: $K_0 = \mathrm{diag}(1, \sqrt{1-p})$, $K_1 = \mathrm{diag}(0, \sqrt{p})$.
    For $d>2$: $d$ Kraus operators, each projecting onto one basis state scaled appropriately.
    """

    @staticmethod
    def ops(d: int, p: float) -> List[pt.Tensor]:
        """
        Return Kraus operators for phase damping on a $d$-level system.

        $K_0 = \\mathrm{diag}(1, \\sqrt{1-p}, \\dots, \\sqrt{1-p})$ and
        $K_j = \\sqrt{p}\\,|j\\rangle\\langle j|$ for $j = 1, \\dots, d-1$.
        """
        K0 = pt.zeros((d, d), dtype=C128)
        K0[0, 0] = 1.0
        for j in range(1, d):
            K0[j, j] = (1 - p) ** 0.5

        result = [K0]
        for j in range(1, d):
            Kj = pt.zeros((d, d), dtype=C128)
            Kj[j, j] = p**0.5
            result.append(Kj)

        return result


class Reset:
    """
    Reset channel Kraus operators for a $d$-level system.

    Collapses the qudit to $|0\\rangle$ with probability $p$:
    $\\Phi(\\rho) = (1-p)\\rho + p\\,|0\\rangle\\langle 0|\\,\\mathrm{Tr}(\\rho)$.
    """

    @staticmethod
    def ops(d: int, p: float) -> List[pt.Tensor]:
        """
        Return Kraus operators for the reset channel.

        $K_j = \\sqrt{p}\\,|0\\rangle\\langle j|$ for $j = 0, \\dots, d-1$
        and $K_d = \\sqrt{1-p}\\,I$.
        """
        result = []
        for j in range(d):
            Kj = pt.zeros((d, d), dtype=C128)
            Kj[0, j] = p**0.5
            result.append(Kj)

        result.append((1 - p) ** 0.5 * pt.eye(d, dtype=C128))

        return result


class ThermalRelax:
    """
    Thermal relaxation channel Kraus operators for a qubit ($d=2$ only).

    Two-parameter model combining $T_1$ energy decay and $T_2$ dephasing.
    $T_1$ is the relaxation time, $T_2$ is the dephasing time, $t$ is the gate time.
    Requires $T_2 \leq 2 T_1$.
    """

    @staticmethod
    def ops(T1: float, T2: float, t: float) -> List[pt.Tensor]:
        """
        Return 4 Kraus operators for thermal relaxation on a qubit.

        Decomposes combined $T_1$/$T_2$ dynamics into amplitude decay ($p_1$) and
        pure dephasing ($p_\\phi$) components:
        $p_1 = 1 - e^{-t/T_1}$,
        $p_\\phi = (1 - e^{-t/T_2} / e^{-t/(2T_1)}) / 2$.
        """
        import math

        e1 = math.exp(-t / T1)
        e2 = math.exp(-t / T2)

        p1 = 1 - e1
        lam = e2 / math.sqrt(e1)
        p_phi = (1 - lam) / 2

        K0 = pt.tensor(
            [[math.sqrt(1 - p_phi), 0.0], [0.0, math.sqrt((1 - p1) * (1 - p_phi))]],
            dtype=C128,
        )
        K1 = pt.tensor(
            [[0.0, math.sqrt(p1 * (1 - p_phi))], [0.0, 0.0]],
            dtype=C128,
        )
        K2 = pt.tensor(
            [[math.sqrt(p_phi), 0.0], [0.0, -math.sqrt((1 - p1) * p_phi)]],
            dtype=C128,
        )
        K3 = pt.tensor(
            [[0.0, math.sqrt(p1 * p_phi)], [0.0, 0.0]],
            dtype=C128,
        )

        return [K0, K1, K2, K3]
