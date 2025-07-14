from scipy.linalg import logm, fractional_matrix_power, svdvals
from typing import List, Union
import numpy as np
from qudit.tools.metrics import Fidelity


class Distance:
    @staticmethod
    def relative_entropy(
        rho: np.ndarray, sigma: np.ndarray, base: float = 2.0
    ) -> float:
        rho = np.outer(rho, rho.conj()) if rho.ndim == 1 else rho

        sigma = np.outer(sigma, sigma.conj()) if sigma.ndim == 1 else sigma

        eps = 1e-12
        rho += eps * np.eye(rho.shape[0])
        sigma += eps * np.eye(sigma.shape[0])

        log_rho = logm(rho)
        log_sigma = logm(sigma)
        delta_log = log_rho - log_sigma

        result = np.trace(rho @ delta_log).real  # ensured
        return float(result / np.log(base))

    @staticmethod
    def bures(rho: np.ndarray, sigma: np.ndarray) -> float:

        rho = np.outer(rho, rho.conj()) if rho.ndim == 1 else rho

        sigma = np.outer(sigma, sigma.conj()) if sigma.ndim == 1 else sigma

        bures_distance = np.sqrt(2 - 2 * (Fidelity.default(rho, sigma)) ** 0.5)

        return float(bures_distance)

    @staticmethod
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

        return 0.5 * np.trace(svdvals(rho - sigma)).real
