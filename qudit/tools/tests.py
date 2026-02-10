from numpy import linalg as LA
import numpy as np


def PPT(rho: np.ndarray, sub: int) -> bool:
    """
    Peres–Horodecki (PPT) separability test for a bipartite density matrix.

    Performs a blockwise partial transpose on subsystem blocks of size `sub` and checks
    positivity: $\\rho^{T_B} \succeq 0$.

    Note: current implementation overrides `sub` to 3.
    """
    side = rho.shape[0]
    sub = 3
    if side % sub != 0:
        raise ValueError(f"Matrix side ({side}) not divisible by sub ({sub})")

    mat0 = rho.copy()
    for i in range(0, mat0.shape[0], sub):
        for j in range(0, mat0.shape[1], sub):
            mat0[i : i + sub, j : j + sub] = mat0[i : i + sub, j : j + sub].T

    return bool(np.all(np.linalg.eigvals(mat0) >= 0))


class Space:
    """
    Linear-algebra helpers for vector spaces and bipartite decompositions.
    """

    @staticmethod
    def gramSchmidt(vectors: np.ndarray) -> np.ndarray:
        """
        Gram-Schmidt orthonormalization.

        Given vectors $\{v_i\}$, constructs an orthonormal set $\{u_i\}$ spanning the same
        subspace (dropping near-zero residuals).
        """
        ortho = []
        for v in vectors:
            w = v - sum(np.dot(v, np.conj(u)) * u for u in ortho)
            if LA.norm(w) > 1e-8:
                ortho.append(w / LA.norm(w))

        return np.array(ortho)

    @staticmethod
    def schmidtDecompose(state: np.ndarray) -> list:
        """
        Schmidt decomposition via SVD.

        Treats `state` as a bipartite coefficient matrix $\Psi$ and returns singular triplets
        $(\lambda_k, |u_k\\rangle, |v_k\\rangle)$ with $\Psi = \sum_k \lambda_k |u_k\\rangle\langle v_k|$.
        """
        U, D, V = LA.svd(state)
        dims = int(np.min(state.shape))

        return sorted(
            [(D[k], U[:, k], V.T[:, k]) for k in range(dims)],
            key=lambda dec: dec[0],
            reverse=True,
        )

    @staticmethod
    def schmidtRank(mat: np.ndarray) -> int:
        """
        Schmidt rank (matrix rank) of a bipartite coefficient matrix.
        """
        return int(LA.matrix_rank(mat))
