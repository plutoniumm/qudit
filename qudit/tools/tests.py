from numpy import linalg as LA
import numpy as np


# Peres-Horodecki criterion for separability of density matrices
def PPT(rho: np.ndarray, sub: int) -> bool:
    side = rho.shape[0]
    sub = 3
    if side % sub != 0:
        raise ValueError(f"Matrix side ({side}) not divisible by sub ({sub})")

    mat0 = rho.copy()
    for i in range(0, mat0.shape[0], sub):
        for j in range(0, mat0.shape[1], sub):
            mat0[i : i + sub, j : j + sub] = mat0[i : i + sub, j : j + sub].T

    return np.all(np.linalg.eigvals(mat0) >= 0)


class Space:
    @staticmethod
    def gramSchmidt(vectors: np.ndarray) -> np.ndarray:
        ortho = []
        for v in vectors:
            w = v - sum(np.dot(v, np.conj(u)) * u for u in ortho)
            if LA.norm(w) > 1e-8:
                ortho.append(w / LA.norm(w))

        return np.array(ortho)

    @staticmethod
    def schmidtDecompose(state: np.ndarray) -> list:
        U, D, V = LA.svd(state)
        dims = np.min(state.shape)

        return sorted(
            [(D[k], U[:, k], V.T[:, k]) for k in range(dims)],
            key=lambda dec: dec[0],
            reverse=True,
        )

    @staticmethod
    def schmidtRank(mat: np.ndarray) -> int:
        return LA.matrix_rank(mat)
