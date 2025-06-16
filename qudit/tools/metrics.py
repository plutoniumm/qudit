from scipy.linalg import logm, fractional_matrix_power
from typing import List, Union
import numpy as np



def _partial_trace(rho: np.ndarray, dA: int, dB: int, keep: str = "A") -> np.ndarray:
        if keep == "A":
            return np.einsum("ij->i", rho.reshape(dA, dB, dA, dB)).reshape(dA, dA)
        elif keep == "B":
            return np.einsum("ij->j", rho.reshape(dA, dB, dA, dB)).reshape(dB, dB)
        else:
            raise ValueError("keep must be 'A' or 'B'")

def _partial_transpose(rho: np.ndarray, dim_A: int, dim_B: int) -> np.ndarray:
        assert rho.shape == (dim_A * dim_B, dim_A * dim_B)
        rho = rho.reshape(dim_A, dim_B, dim_A, dim_B)
        rho_pt = np.transpose(rho, axes=(0, 2, 1, 3))
        return rho_pt.reshape(dim_A * dim_B, dim_A * dim_B)
    
class Fidelity:

    @staticmethod
    @staticmethod
    def default(rho: np.ndarray, sigma: np.ndarray) -> float:
        if rho.ndim == 1 and sigma.ndim == 1:
            return float(np.abs(np.vdot(rho, sigma)) ** 2)

        if rho.ndim == 1:
            rho = np.outer(rho, rho.conj())
        if sigma.ndim == 1:
            sigma = np.outer(sigma, sigma.conj())

        sqrt_rho = fractional_matrix_power(rho, 0.5)
        inner = sqrt_rho @ sigma @ sqrt_rho
        fidelity = (np.trace(fractional_matrix_power(inner, 0.5))) ** 2
        return float(np.clip(np.real(fidelity)), 0, 1)
        fidelity = (np.trace(fractional_matrix_power(inner, 0.5))) ** 2
        return float(np.real(fidelity))

    @staticmethod
    @staticmethod
    def channel(
        kraus: List[Union[np.ndarray, List[float]]], rho: np.ndarray
    ) -> np.ndarray:
        if rho.ndim == 1:
            rho = np.outer(rho, rho.conj())

        d_1, d_2 = kraus[0].shape
        if rho.shape != (d_2, d_2):
            raise ValueError(
                f"Incompatible shape: expected {(d_2, d_2)}, got {rho.shape}"
            )

        rho_out = np.zeros((d_1, d_1), dtype=complex)
        for K in kraus:
            rho_out += K @ rho @ K.conj().T

        return rho_out

    # TODO: is ndim enough? or do we need to check for square?
    @staticmethod
    @staticmethod
    def entanglement(rho: np.ndarray, kraus_ops: List[np.ndarray]) -> float:
        d = rho.shape[0]
        assert rho.shape == (d, d), "rho must be a square matrix"
        for K in kraus_ops:
            assert K.shape == (d, d), "Each Kraus operator must be of shape (d, d)"

        F_e = 0.0
        for K in kraus_ops:
            term = np.trace(rho @ K)
            F_e += np.abs(term) ** 2

        return F_e

    @staticmethod
    def cafaro(kraus_ops: List[np.ndarray]) -> float:
        N = kraus_ops[0].shape[0]
        for K in kraus_ops:
            assert K.shape == (N, N)

        F_e = sum(np.abs(np.trace(K)) ** 2 for K in kraus_ops)
        return F_e / (N**2)

    @staticmethod
    def negativity(rho: np.ndarray, dim_A: int, dim_B: int) -> float:
        rho_reshaped = rho.reshape(dim_A, dim_B, dim_A, dim_B)
        rho_pt = np.transpose(rho_reshaped, axes=(0, 3, 2, 1))
        rho_pt = rho_pt.reshape(dim_A * dim_B, dim_A * dim_B)
        singular_values = np.linalg.svd(rho_pt, compute_uv=False)
        trace_norm = np.sum(singular_values)
        return (trace_norm - 1) / 2


class Entropy:

    @staticmethod
    def default(*args):
        return Entropy.neumann(*args)

    @staticmethod
    def tsallis(rho: np.ndarray, q: float = 2.0, base: float = 2.0) -> float:
        if q == 1:
            return Entropy.neumann(rho, base=base)
        rho = np.outer(rho, rho.conj()) if rho.ndim == 1 else rho

        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        return (1 - np.sum(eigenvalues**q)) / (q - 1)

    @staticmethod
    def shannon(probs: np.ndarray, base: float = 2.0) -> float:
        probs = probs[probs > 1e-12]
        return -np.sum(probs * np.log(probs) / np.log(base))

    @staticmethod
    def renyi(rho: np.ndarray, alpha: float = 2.0, base: float = 2.0) -> float:
        if alpha == 1:
            return Entropy.neumann(
                rho, base=base
            )  # renyi entropy with alpha=1 is the same as von Neumann entropy
        rho = np.outer(rho, rho.conj()) if rho.ndim == 1 else rho

        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        return np.log(np.sum(eigenvalues**alpha)) / ((1 - alpha) * np.log(base))

    @staticmethod
    def hartley(probs: np.ndarray, base: float = 2.0) -> float:
        support_size = np.count_nonzero(probs > 1e-12)
        return np.log(support_size) / np.log(base)

    @staticmethod
    def neumann(rho: np.ndarray, base: float = 2.0) -> float:
        rho = np.outer(rho, rho.conj()) if rho.ndim == 1 else rho

        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        return -np.sum(eigenvalues * np.log(eigenvalues) / np.log(base))

    @staticmethod
    def unified(
        rho: np.ndarray, q: float = 2.0, alpha: float = 2.0, base: float = 2.0
    ) -> float:
        rho = np.outer(rho, rho.conj()) if rho.ndim == 1 else rho

        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        s = np.sum(eigenvalues**alpha)

        if abs(q - 1.0) < 1e-8:
            return np.log(s) / ((1 - alpha) * np.log(base))  # renyi
        elif abs(alpha - 1.0) < 1e-8:
            return (1 - np.sum(eigenvalues**q)) / ((q - 1))  # tsallis
        else:
            return ((s ** ((1 - q) / (1 - alpha))) - 1) / (1 - q)

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

        result = np.trace(rho @ delta_log).real
        return float(result / np.log(base))

    @staticmethod
    def conditional_entropy(rho: np.ndarray, dA: int, dB: int) -> float:
        
        rho_A = _partial_trace(rho, dA, dB, keep="A")
        S_A = Entropy.default(rho_A)
        S_AB = Entropy.default(rho)

        return S_AB - S_A


class Info:

    @staticmethod
    def conditional_entropy(
        rho: np.ndarray, dA: int, dB: int, true_case: bool = True
    ) -> float:

        if true_case:

            def _projection(d: int):
                return [np.outer(b, b) for b in np.eye(d)]
            projectors = _projection(dA)
            S_cond = 0
            for P in projectors:
                Pi = np.kron(P, np.eye(dB))
                prob = np.trace(Pi @ rho)
                if prob > 1e-12:
                    rho_cond = Pi @ rho @ Pi / prob
                    rho_B = _partial_trace(rho_cond, dA, dB, keep="B")
                    S_cond += prob * Entropy.default(rho_B)
            return S_cond
        else:

            assert rho.shape == (dA * dB, dA * dB)
            rho_A = _partial_trace(rho, dA, dB, keep="A")
            S_A = Entropy.default(rho_A)
            S_AB = Entropy.default(rho)
            return S_AB - S_A

    @staticmethod
    def mutual(rho: np.ndarray, dA: int, dB: int) -> float:
        assert rho.shape == (dA * dB, dA * dB)

        rho_A = _partial_trace(rho, dA, dB, keep="A")
        rho_B = _partial_trace(rho, dA, dB, keep="B")
        S_A = Entropy.default(rho_A)
        S_B = Entropy.default(rho_B)
        S_AB = Entropy.default(rho)
        return S_A + S_B - S_AB

    @staticmethod
    def coherent(rho_AB: np.ndarray, dA: int, dB: int) -> float:
        assert rho_AB.shape == (dA * dB, dA * dB), "rho must be of shape (dA*dB, dA*dB)"
        rho_B = _partial_trace(rho_AB, dA, dB, keep="B")
        S_B = Entropy.default(rho_B)
        S_AB = Entropy.default(rho_AB)

        return S_B - S_AB
