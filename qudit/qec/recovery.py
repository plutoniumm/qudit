from ..noise import Channel
from ..circuit import Gate
from typing import List
import torch as pt
import math

C64 = pt.complex64


class Recovery:
    """
    Construct common approximate/analytic recovery maps for a code subspace
    using PyTorch and localized qudit Gates.
    """

    @staticmethod
    def _apply(
        word: List[Gate], total_dim: int, device: pt.device, dtype: pt.dtype
    ) -> pt.Tensor:
        """
        Materialize the full $d^n \\times d^n$ matrix for a Kraus word by
        passing an identity through the localized `_left()` calls.
        """
        Ek_mat = pt.eye(total_dim, dtype=dtype, device=device)

        for g in word:
            Ek_mat = g._left(Ek_mat)

        return Ek_mat

    @staticmethod
    def leung(channel: Channel, codes: List[pt.Tensor]) -> Channel:
        device = codes[0].device
        dtype = codes[0].dtype
        first_gate = channel.ops[0][0]
        n = first_gate.wires
        d = first_gate.dims[0] if hasattr(first_gate, "dims") else 2
        total_dim = codes[0].numel()

        P = pt.zeros((total_dim, total_dim), dtype=dtype, device=device)
        for c in codes:
            c_col = c.view(total_dim, 1)
            P += c_col @ c_col.conj().T

        R_gates = []
        for i, word in enumerate(channel.ops):
            Ek_mat = Recovery._apply(word, total_dim, device, dtype)

            A = Ek_mat @ P
            U, S, Vh = pt.linalg.svd(A, full_matrices=False)
            Uk = U @ Vh

            Rk_mat = P @ Uk.conj().T

            rg = Gate(Rk_mat, index=list(range(n)), wires=n, dim=d, name=f"R_leung_{i}")
            R_gates.append([rg])

        return Channel(R_gates)

    @staticmethod
    def cafaro(channel: Channel, codes: List[pt.Tensor]) -> Channel:
        device = codes[0].device
        dtype = codes[0].dtype
        first_gate = channel.ops[0][0]
        n = first_gate.wires
        d = first_gate.dims[0] if hasattr(first_gate, "dims") else 2
        total_dim = codes[0].numel()

        R_gates = []
        for i, word in enumerate(channel.ops):
            Ek_mat = Recovery._apply(word, total_dim, device, dtype)
            Rk_mat = pt.zeros((total_dim, total_dim), dtype=dtype, device=device)

            for c in codes:
                c_col = c.view(total_dim, 1)
                overlap = (
                    (c_col.conj().T @ Ek_mat.conj().T @ Ek_mat @ c_col).squeeze().real
                )

                if overlap > 1e-12:
                    proj = c_col @ c_col.conj().T
                    Rk_mat += (proj @ Ek_mat.conj().T) / math.sqrt(overlap)

            rg = Gate(
                Rk_mat, index=list(range(n)), wires=n, dim=d, name=f"R_cafaro_{i}"
            )
            R_gates.append([rg])

        return Channel(R_gates)

    @staticmethod
    def petz(channel: Channel, codes: List[pt.Tensor]) -> Channel:
        device = codes[0].device
        first_gate = channel.ops[0][0]
        n = first_gate.wires
        d = first_gate.dims[0] if hasattr(first_gate, "dims") else 2
        total_dim = codes[0].numel()

        P = pt.zeros((total_dim, total_dim), dtype=C64, device=device)
        for c in codes:
            c_col = c.view(total_dim, 1)
            P += c_col @ c_col.conj().T

        E_P = channel.run(P)

        L, V = pt.linalg.eigh(E_P)
        L_rinv = pt.zeros_like(L)
        mask = L > 1e-10
        L_rinv[mask] = 1.0 / pt.sqrt(L[mask])

        L, V = L.to(C64), V.to(C64)
        L_rinv = L_rinv.to(C64)
        norm = V @ pt.diag(L_rinv) @ V.conj().T

        R_gates = []
        for i, word in enumerate(channel.ops):
            Ek_mat = Recovery._apply(word, total_dim, device, C64)
            Rk_mat = P @ Ek_mat.conj().T @ norm

            rg = Gate(Rk_mat, index=list(range(n)), wires=n, dim=d, name=f"R_petz_{i}")
            R_gates.append([rg])

        return Channel(R_gates)
