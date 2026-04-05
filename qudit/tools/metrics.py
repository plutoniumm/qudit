from scipy.linalg import logm, fractional_matrix_power, svdvals
from qudit.utils import partial
from typing import List, Union, Any, Optional
from numpy import linalg as LA
import numpy as np

MD = LA.multi_dot


class Fidelity:
    """
    Fidelity-like functionals for states and channels.
    """

    @staticmethod
    def default(rho: np.ndarray, sigma: np.ndarray) -> float:
        """
        Uhlmann fidelity $F(\\rho,\sigma)$.

        For pure statevectors $|\psi\\rangle, |\phi\\rangle$: $F = |\langle\psi|\phi\\rangle|^2$.
        For density matrices: $F = \\big(\mathrm{Tr}\sqrt{\sqrt{\\rho}\,\sigma\,\sqrt{\\rho}}\\big)^2$.
        """
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

    @staticmethod
    def channel(
        kraus: List[Union[np.ndarray, List[float]]], rho: np.ndarray
    ) -> np.ndarray:
        """
        Apply a quantum channel given by Kraus operators.

        Computes $\mathcal{E}(\\rho)=\sum_k K_k\,\\rho\,K_k^\dagger$.
        """
        rho_out = np.zeros_like(rho, dtype=np.complex128)
        for K in kraus:
            K_arr = np.asarray(K)
            rho_out += K_arr @ rho @ K_arr.conj().T

        return rho_out

    @staticmethod
    def entanglement(
        R_kraus: List[np.ndarray], E_kraus: List[np.ndarray], codes: List[np.ndarray]
    ) -> float:
        """
        Entanglement fidelity for an encode-noise-recovery pipeline.

        Builds a purification $|QR\\rangle$ from `codes`, applies noise $\mathcal{E}$ and recovery
        $\mathcal{R}$ via Kraus sets, and returns $\langle QR|\\rho'|QR\\rangle$.
        """
        l = len(codes)
        R = np.eye(l)

        QR = (1 / np.sqrt(l)) * sum([np.kron(codes[i], R[i]) for i in range(l)])

        rho = np.outer(QR, QR.conj().T)

        Eks = [np.kron(Ek, R) for Ek in E_kraus]
        Rks = [np.kron(Rk, R) for Rk in R_kraus]

        rho_new = sum([MD([Ek, rho, Ek.conj().T]) for Ek in Eks])
        rho_new = sum([MD([Rk, rho_new, Rk.conj().T]) for Rk in Rks])

        rho_new /= np.trace(rho_new)

        fid = np.dot(QR.conj().T, np.dot(rho_new, QR))

        return np.abs(fid)

    @staticmethod
    def bare_qubit(
        R_kraus: List[np.ndarray], E_kraus: List[np.ndarray], state: np.ndarray
    ) -> float:
        """
        Fidelity for a bare qubit under noise and recovery.

        The noise channel and recovery are applied to the state via Kraus operators, and the fidelity is computed as $\langle\psi|\\rho'|\psi\\rangle$ where $\\rho' = \\sum_k R_k \\, \\sum_l E_l \\, |\\psi\\rangle\\langle\\psi| \\, E_l^\dagger \\, R_k^\dagger$.
        """

        rho = np.outer(state, state.conj().T)

        rho = sum([MD([Ek, rho, Ek.conj().T]) for Ek in E_kraus])
        rho = sum([MD([Rk, rho, Rk.conj().T]) for Rk in R_kraus])

        return np.abs(MD([state.conj().T, rho, state]))

    @staticmethod
    def cafaro(
        R_kraus: List[np.ndarray], E_kraus: List[np.ndarray], codes: List[np.ndarray]
    ) -> float:
        """
        Cafaro-style entanglement fidelity proxy for a channel.

        Uses $F_e = \\frac{1}{d|\mathcal{C}|} \sum_{k,l} |\sum_i \langle i|R_k E_l|i\\rangle|^2$ where $\{|i\\rangle\}$ are the codewords.
        """
        f_ent = 0.0
        for Al in E_kraus:
            for Rk in R_kraus:
                f_ent += (
                    np.sum([MD([state.conj().T, Rk, Al, state]) for state in codes])
                    ** 2
                )

        return np.real(f_ent / (len(codes) ** 2))

    @staticmethod
    def negativity(rho: np.ndarray, dim_A: int, dim_B: int) -> float:
        """
        Negativity of a bipartite state via the partial transpose.

        Computes $\mathcal{N}(\\rho)=\\tfrac{\|\\rho^{T_B}\|_1-1}{2}$ using the trace norm.
        """
        rho_reshaped = rho.reshape(dim_A, dim_B, dim_A, dim_B)
        rho_pt = np.transpose(rho_reshaped, axes=(0, 3, 2, 1))
        rho_pt = rho_pt.reshape(dim_A * dim_B, dim_A * dim_B)
        singular_values = np.linalg.svd(rho_pt, compute_uv=False)
        trace_norm = np.sum(singular_values)

        return float((trace_norm - 1) / 2)


class Entropy:
    """
    Common entropy functionals for probability vectors and density matrices.
    """

    @staticmethod
    def default(*args: Any) -> float:
        """
        Default entropy (alias for von Neumann entropy).
        """
        return float(Entropy.neumann(*args))

    @staticmethod
    def tsallis(rho: np.ndarray, q: float = 2.0, base: float = 2.0) -> float:
        """
        Tsallis entropy $S_q$ for a state (typically density matrix).

        For eigenvalues $\{\lambda_i\}$: $S_q = \\frac{1-\sum_i \lambda_i^q}{q-1}$.
        """
        if q == 1:
            return float(Entropy.neumann(rho, base=base))

        rho = np.outer(rho, rho.conj()) if rho.ndim == 1 else rho

        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]

        return float((1 - np.sum(eigenvalues**q)) / (q - 1))

    @staticmethod
    def shannon(probs: np.ndarray, base: float = 2.0) -> float:
        """
        Shannon entropy $H(p)=-\sum i p_i \log p_i$ for a probability vector.
        """
        probs = probs[probs > 1e-12]

        return float(-np.sum(probs * np.log(probs) / np.log(base)))

    @staticmethod
    def renyi(rho: np.ndarray, alpha: float = 2.0, base: float = 2.0) -> float:
        """
        Renyi entropy $S_\\alpha$ for a density matrix.

        For eigenvalues $\{\lambda_i\}$: $S_\\alpha = \\frac{1}{1-\\alpha}\log\sum i \lambda_i^\\alpha$.
        """
        if alpha == 1:
            return float(Entropy.neumann(rho, base=base))

        rho = np.outer(rho, rho.conj()) if rho.ndim == 1 else rho

        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]

        return float(np.log(np.sum(eigenvalues**alpha)) / ((1 - alpha) * np.log(base)))

    @staticmethod
    def hartley(probs: np.ndarray, base: float = 2.0) -> float:
        """
        Hartley entropy $H_0=\log |\mathrm{supp}(p)|$ for a probability vector.
        """
        support_size = np.count_nonzero(probs > 1e-12)

        return float(np.log(support_size) / np.log(base))

    @staticmethod
    def neumann(rho: np.ndarray, base: float = 2.0) -> float:
        """
        von Neumann entropy $S(\\rho)=-\mathrm{Tr}(\\rho\log\\rho)$.
        """
        rho = np.outer(rho, rho.conj()) if rho.ndim == 1 else rho

        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]

        return float(-np.sum(eigenvalues * np.log(eigenvalues) / np.log(base)))

    @staticmethod
    def unified(
        rho: np.ndarray, q: float = 2.0, alpha: float = 2.0, base: float = 2.0
    ) -> float:
        """
        Unified $(q,\\alpha)$-entropy family (interpolates Tsallis/Renyi cases).
        """
        rho = np.outer(rho, rho.conj()) if rho.ndim == 1 else rho

        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        s = np.sum(eigenvalues**alpha)

        if abs(q - 1.0) < 1e-8:
            return float(np.log(s) / ((1 - alpha) * np.log(base)))
        elif abs(alpha - 1.0) < 1e-8:
            return float((1 - np.sum(eigenvalues**q)) / (q - 1))
        else:
            return float(((s ** ((1 - q) / (1 - alpha))) - 1) / (1 - q))

    @staticmethod
    def relative(rho: np.ndarray, sigma: np.ndarray, base: float = 2.0) -> float:
        """
        Quantum relative entropy $D(\\rho\|\sigma)=\mathrm{Tr}[\\rho(\log\\rho-\log\sigma)]$.
        """
        rho = np.outer(rho, rho.conj()) if rho.ndim == 1 else rho

        sigma = np.outer(sigma, sigma.conj()) if sigma.ndim == 1 else sigma

        eps = 1e-12
        rho += eps * np.eye(rho.shape[0])
        sigma += eps * np.eye(sigma.shape[0])

        log_rho = np.asarray(logm(rho))
        log_sigma = np.asarray(logm(sigma))
        delta_log = log_rho - log_sigma

        result = np.trace(rho @ delta_log).real

        return float(result / np.log(base))

    @staticmethod
    def conditional(rho: np.ndarray, dA: int, dB: int) -> float:
        """
        Conditional entropy $S(A|B)=S(AB)-S(A)$ for a bipartite state.
        """
        assert rho.shape == (
            dA * dB,
            dA * dB,
        ), "Input must be a square matrix of shape (dA*dB, dA*dB)"

        rho_B = partial.trace(rho, dA, dB, keep="B")
        S_B = Entropy.default(rho_B)
        S_AB = Entropy.default(rho)

        return S_AB - S_B


class Info:
    """
    Information-theoretic quantities derived from entropies.
    """

    @staticmethod
    def conditional(rho: np.ndarray, dA: int, dB: int, true_case: bool = True) -> float:
        """
        Conditional entropy (two conventions; controlled by `true_case`).

        - If `true_case=True`, computes the measurement-induced conditional entropy on B given a
          computational basis projective measurement on A.
        - Otherwise returns $S(AB)-S(A)$.
        """
        if true_case:
            d = dA
            projectors = [np.outer(b, b) for b in np.eye(d)]
            S_cond = 0

            for P in projectors:
                Pi = np.kron(P, np.eye(dB))
                prob = np.trace(Pi @ rho)
                if prob > 1e-12:
                    rho_cond = Pi @ rho @ Pi / prob
                    rho_B = partial.trace(rho_cond, dA, dB, keep="B")
                    S_cond += prob * Entropy.default(rho_B)

            return S_cond
        else:
            assert rho.shape == (dA * dB, dA * dB), f"rho must be ({dA*dB},{dA*dB}), got {rho.shape}"

            rho_B = partial.trace(rho, dA, dB, keep="B")
            S_B = Entropy.default(rho_B)
            S_AB = Entropy.default(rho)

            return S_AB - S_B

    @staticmethod
    def mutual(rho: np.ndarray, dA: int, dB: int) -> float:
        """
        Quantum mutual information $I(A:B)=S(A)+S(B)-S(AB)$.
        """

        assert rho.shape == (dA * dB, dA * dB), f"rho must be ({dA*dB},{dA*dB}), got {rho.shape}"

        rho_A = partial.trace(rho, dA, dB, keep="A")
        rho_B = partial.trace(rho, dA, dB, keep="B")
        S_A = Entropy.default(rho_A)
        S_B = Entropy.default(rho_B)
        S_AB = Entropy.default(rho)

        return S_A + S_B - S_AB

    @staticmethod
    def coherent(rho_AB: np.ndarray, dA: int, dB: int) -> float:
        """
        Coherent information $I_c(A\\rangle B)=S(B)-S(AB)$.
        """
        assert rho_AB.shape == (
            dA * dB,
            dA * dB,
        ), f"expected rho ({dA*dB}, {dA*dB}), got {rho_AB.shape}"

        rho_B = partial.trace(rho_AB, dA, dB, keep="B")
        S_B = Entropy.default(rho_B)
        S_AB = Entropy.default(rho_AB)

        return S_B - S_AB


class Distance:
    """
    Distances/divergences between quantum states.
    """

    @staticmethod
    def relative_entropy(
        rho: np.ndarray, sigma: np.ndarray, base: float = 2.0
    ) -> float:
        """
        Relative entropy distance $D(\\rho\|\sigma)$ (same as `Entropy.relative`).
        """
        rho = np.outer(rho, rho.conj()) if rho.ndim == 1 else rho

        sigma = np.outer(sigma, sigma.conj()) if sigma.ndim == 1 else sigma

        eps = 1e-12
        rho += eps * np.eye(rho.shape[0])
        sigma += eps * np.eye(sigma.shape[0])

        log_rho = np.asarray(logm(rho))
        log_sigma = np.asarray(logm(sigma))
        delta_log = log_rho - log_sigma

        result = np.trace(rho @ delta_log).real

        return float(result / np.log(base))

    @staticmethod
    def bures(rho: np.ndarray, sigma: np.ndarray) -> float:
        """
        Bures distance $D_B(\\rho,\sigma)=\sqrt{2-2\sqrt{F(\\rho,\sigma)}}$.
        """
        rho = np.outer(rho, rho.conj()) if rho.ndim == 1 else rho

        sigma = np.outer(sigma, sigma.conj()) if sigma.ndim == 1 else sigma

        bures_distance = np.sqrt(2 - 2 * (Fidelity.default(rho, sigma)) ** 0.5)

        return float(bures_distance)

    @staticmethod
    def jensen_shannon(rho: np.ndarray, sigma: np.ndarray, base: float = 2.0) -> float:
        """
        Quantum Jensen-Shannon divergence (symmetric, smoothed version of relative entropy).

        Uses $\\tfrac12(D(\\rho\|m)+D(\sigma\|m))$ with $m=(\\rho+\sigma)/2$.
        """
        rho = np.outer(rho, rho.conj()) if rho.ndim == 1 else rho
        sigma = np.outer(sigma, sigma.conj()) if sigma.ndim == 1 else sigma

        m = 0.5 * (rho + sigma)

        return 0.5 * (
            Distance.relative_entropy(rho, m, base)
            + Distance.relative_entropy(sigma, m, base)
        )

    @staticmethod
    def trace(rho: np.ndarray, sigma: np.ndarray) -> float:
        """
        Trace distance $\\tfrac12\|\\rho-\sigma\|_1$ via singular values.
        """
        rho = np.outer(rho, rho.conj()) if rho.ndim == 1 else rho
        sigma = np.outer(sigma, sigma.conj()) if sigma.ndim == 1 else sigma

        return float(0.5 * np.sum(svdvals(rho - sigma)))
