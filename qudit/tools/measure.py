from scipy.linalg import logm, fractional_matrix_power, svdvals
from typing import List, Union,fractional_matrix_power,svdvals
from typing import List, Union
import numpy as np

@staticmethod
def density(matrix: np.ndarray) -> np.ndarray:
    return np.outer(matrix, matrix.conj()) if matrix.ndim == 1 else matrix


class Distance:
    @staticmethod
    def relative_entropy(
        rho: np.ndarray, sigma: np.ndarray, base: float = 2.0
    ) -> float:
        rho = np.outer(rho, rho.conj()) if rho.ndim == 1 else rho

        sigma = np.outer(sigma, sigma.conj()) if sigma.ndim == 1 else sigma
        rho = np.outer(rho, rho.conj()) if rho.ndim == 1 else rho

        sigma = np.outer(sigma, sigma.conj()) if sigma.ndim == 1 else sigma

        eps = 1e-12
        rho += eps * np.eye(rho.shape[0])
        sigma += eps * np.eye(sigma.shape[0])

        log_rho = logm(rho)
        log_sigma = logm(sigma)
        delta_log = log_rho - log_sigma

        result = np.trace(rho @ delta_log).real  #  ensured
        return float(result / np.log(base))


    @staticmethod
    def bures(rho: np.ndarray, sigma: np.ndarray) -> float:

        rho = np.outer(rho, rho.conj()) if rho.ndim == 1 else rho

        sigma = np.outer(sigma, sigma.conj()) if sigma.ndim == 1 else sigma

        bures_distance = np.sqrt(2 - 2 * (Fidelity.default(rho, sigma)) ** 0.5)


        rho = np.outer(rho, rho.conj()) if rho.ndim == 1 else rho

        sigma = np.outer(sigma, sigma.conj()) if sigma.ndim == 1 else sigma

        if rho.ndim == 1 and sigma.ndim == 1:
            return float(np.abs(np.vdot(rho, sigma)) ** 2)

        if rho.ndim == 1:
            rho = np.outer(rho, rho.conj())
        if sigma.ndim == 1:
            sigma = np.outer(sigma, sigma.conj())

        sqrt_rho = fractional_matrix_power(rho, 0.5)
        inner = sqrt_rho @ sigma @ sqrt_rho
        fidelity = (np.trace(fractional_matrix_power(inner, 0.5))) ** 2

        bures_distance = np.sqrt(2 - 2 * fidelity) ** 0.5

        return float(bures_distance)


    @staticmethod
    def jensen_shannon(rho: np.ndarray, sigma: np.ndarray, base: float = 2.0) -> float:
        rho = np.outer(rho, rho.conj()) if rho.ndim == 1 else rho
        sigma = np.outer(sigma, sigma.conj()) if sigma.ndim == 1 else sigma
    def jensen_shannon(rho: np.ndarray, sigma: np.ndarray, base: float = 2.0) -> float:
        rho = np.outer(rho, rho.conj()) if rho.ndim == 1 else rho
        sigma = np.outer(sigma, sigma.conj()) if sigma.ndim == 1 else sigma

        m = 0.5 * (rho + sigma)
        return 0.5 * (
            Distance.relative_entropy(rho, m, base)
            + Distance.relative_entropy(sigma, m, base)
        )

    @staticmethod
    def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
        rho = np.outer(rho, rho.conj()) if rho.ndim == 1 else rho
        sigma = np.outer(sigma, sigma.conj()) if sigma.ndim == 1 else sigma
        return 0.5 * (
            Distance.relative_entropy(rho, m, base)
            + Distance.relative_entropy(sigma, m, base)
        )

        return 0.5 * np.trace(svdvals(rho - sigma)).real

    @staticmethod
    def bhattacharyya(rho: np.ndarray, sigma: np.ndarray, base: float = 2.0) -> float:
        rho = np.outer(rho, rho.conj()) if rho.ndim == 1 else rho
        sigma = np.outer(sigma, sigma.conj()) if sigma.ndim == 1 else sigma

        if rho.ndim == 1 and sigma.ndim == 1:
            return float(np.abs(np.vdot(rho, sigma)) ** 2)

        if rho.ndim == 1:
            rho = np.outer(rho, rho.conj())
        if sigma.ndim == 1:
            sigma = np.outer(sigma, sigma.conj())

        sqrt_rho = fractional_matrix_power(rho, 0.5)
        inner = sqrt_rho @ sigma @ sqrt_rho
        fidelity = (np.trace(fractional_matrix_power(inner, 0.5))) ** 2
        return float(np.real(fidelity))

    def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
        rho = np.outer(rho, rho.conj()) if rho.ndim == 1 else rho
        sigma = np.outer(sigma, sigma.conj()) if sigma.ndim == 1 else sigma

        return 0.5 * np.trace(svdvals(rho - sigma)).real
