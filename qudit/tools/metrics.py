from scipy.linalg import fractional_matrix_power
from typing import List, Union
from .. import Dit, Psi, In
import numpy as np

def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    # Convert pure states to density matrices if needed
    if rho.ndim == 1:
        rho = np.outer(rho, rho.conj())
    if sigma.ndim == 1:
        sigma = np.outer(sigma, sigma.conj())

    # Validate shapes
    if rho.shape != sigma.shape:
        raise ValueError("rho and sigma must be of the same dimension.")

    # Calculate fidelity
    sqrt_rho = fractional_matrix_power(rho, 0.5)
    inner = sqrt_rho @ sigma @ sqrt_rho
    fidelity = np.trace(fractional_matrix_power(inner, 0.5))
    return float(np.real(fidelity))




def channel(kraus: List[np.ndarray], rho: Union[np.ndarray]) -> np.ndarray:
  
    if rho.ndim == 1:
        rho = np.outer(rho, rho.conj())

    d_1, d_2 = kraus[0].shape
    if rho.shape != (d_2, d_2):
        raise ValueError(f"Incompatible shape: expected {(d_2, d_2)}, got {rho.shape}")

    rho_out = np.zeros((d_1, d_1), dtype=complex)
    for K in kraus:
        rho_out += K @ rho @ K.conj().T

    return rho_out
 


def entanglement_fidelity(rho: np.ndarray, kraus_ops: List[np.ndarray]) -> float:
    
    d = rho.shape[0]
    assert rho.shape == (d, d), "rho must be a square matrix"
    for K in kraus_ops:
        assert K.shape == (d, d), "Each Kraus operator must be of shape (d, d)"

    F_e = 0.0
    for K in kraus_ops:
        term = np.trace(rho @ K.conj().T @ K @ rho)
        F_e += np.real(term)

    return F_e



def partial_transpose(rho, dim_A, dim_B):
   
    rho = rho.reshape((dim_A, dim_B, dim_A, dim_B))
    rho_pt = np.transpose(rho, (0, 3, 2, 1))
    return rho_pt.reshape((dim_A * dim_B, dim_A * dim_B))

def negativity(rho, dim_A, dim_B):
    
    rho_pt = partial_transpose(rho, dim_A, dim_B)
    eigenvalues = np.linalg.eigvalsh(rho_pt)
    return np.sum(np.abs(eigenvalues[eigenvalues < 0]))



def entropy(rho: np.ndarray, base: float = 2) -> float:
    evals = np.linalg.eigvalsh(rho)
    evals = evals[evals > 1e-12] 
    return float(-np.sum(evals * np.log(evals) / np.log(base)))

def partial_trace(rho: np.ndarray, dims: List[int], keep: str = 'A') -> np.ndarray:
    dA, dB = dims
    rho = rho.reshape(dA, dB, dA, dB)
    if keep == 'A':
        return np.trace(rho, axis1=1, axis2=3)
    elif keep == 'B':
        return np.trace(rho, axis1=0, axis2=2)
    else:
        raise ValueError(" Should be 'A' or 'B'")

def random_unitary(d: int) -> np.ndarray:
    Z = np.random.randn(d, d) + 1j * np.random.randn(d, d)
    Q, R = np.linalg.qr(Z)
    D = np.diag(R)
    Q *= D / np.abs(D)
    return Q

def generate_projectors_from_unitary(U: np.ndarray) -> List[np.ndarray]:
    d = U.shape[0]
    projectors = []
    for k in range(d):
        basis_k = np.zeros((d,))
        basis_k[k] = 1
        proj = U @ np.outer(basis_k, basis_k) @ U.conj().T
        projectors.append(proj)
    return projectors

def conditional_entropy(rho_AB: np.ndarray, dims: List[int], trials: int = 50) -> float:
    dA, dB = dims
    I_A = np.eye(dA)
    min_entropy = float('inf')

    for _ in range(trials):
        U = random_unitary(dB)
        Pi_list = generate_projectors_from_unitary(U)
        entropy = 0.0
        for Pi_k in Pi_list:
            M_k = np.kron(I_A, Pi_k)
            rho_k = M_k @ rho_AB @ M_k
            p_k = np.trace(rho_k)
            if p_k > 1e-12:
                rho_k /= p_k
                rho_Ak = partial_trace(rho_k, dims, keep='B')
                entropy += p_k * entropy(rho_Ak)
        min_entropy = min(min_entropy, entropy)
    return min_entropy

def quantum_discord(rho_AB: np.ndarray, dims: List[int], optimize: bool = True, trials: int = 50) -> float:
    rho_A = partial_trace(rho_AB, dims, keep='B')
    rho_B = partial_trace(rho_AB, dims, keep='A')
    S_A = entropy(rho_A)
    S_B = entropy(rho_B)
    S_AB = entropy(rho_AB)
    I_AB = S_A + S_B - S_AB
    S_A_given_B = (
        conditional_entropy(rho_AB, dims, trials)
        if optimize else
        NotImplementedError("Non-optimized version not included")
    )
    return I_AB - (S_A - S_A_given_B)
  
