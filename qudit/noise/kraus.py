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
