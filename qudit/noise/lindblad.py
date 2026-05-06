import torch as pt

C128 = pt.complex128


def _lindblad_term(L_op: pt.Tensor, gamma: float, d_sq: int) -> pt.Tensor:
    d = int(d_sq**0.5)
    I = pt.eye(d, dtype=C128)

    LdL = L_op.conj().T @ L_op

    # vec(LρL†) = (L* ⊗ L) vec(ρ)
    # vec(L†Lρ) = (I ⊗ L†L) vec(ρ)
    # vec(ρL†L) = ((L†L)^T ⊗ I) vec(ρ)  — (L†L)^T = conj(L†L) since L†L is Hermitian
    return gamma * (
        pt.kron(L_op.conj(), L_op)
        - 0.5 * pt.kron(I, LdL)
        - 0.5 * pt.kron(LdL.conj().contiguous(), I)
    )


def lindblad_kraus(
    H: pt.Tensor | None,
    T1: float,
    T2: float,
    t_gate: float,
    d: int,
) -> pt.Tensor:
    assert T2 <= 2 * T1, "T2 must be <= 2*T1"

    d_sq = d * d
    I = pt.eye(d, dtype=C128)
    L = pt.zeros((d_sq, d_sq), dtype=C128)

    gamma1 = 1.0 / T1
    gamma_phi = 1.0 / T2 - 1.0 / (2.0 * T1)

    for k in range(1, d):
        # amplitude damping: |k-1><k|
        Lk = pt.zeros((d, d), dtype=C128)
        Lk[k - 1, k] = 1.0
        L = L + _lindblad_term(Lk, gamma1, d_sq)

    for k in range(1, d):
        # pure dephasing: |k><k|
        Lk = pt.zeros((d, d), dtype=C128)
        Lk[k, k] = 1.0
        L = L + _lindblad_term(Lk, gamma_phi, d_sq)

    if H is not None:
        H = H.to(C128)
        L = L - 1j * (pt.kron(I, H) - pt.kron(H.T.contiguous(), I))

    S = pt.linalg.matrix_exp(L * t_gate)

    # superoperator S to Choi: reshape (d²,d²) -> (d,d,d,d) then permute
    choi = S.reshape(d, d, d, d).permute(0, 2, 1, 3).reshape(d_sq, d_sq)

    evals, evecs = pt.linalg.eigh(choi)

    tol = 1e-10
    mask = evals.real > tol
    lam = evals[mask]
    vecs = evecs[:, mask]

    kraus = (lam.real.sqrt().unsqueeze(0) * vecs).T.reshape(-1, d, d)

    return kraus.to(C128)
