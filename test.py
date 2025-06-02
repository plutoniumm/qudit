from qudit import *

def Qubit_Bell() -> Dit:
  D = DGate(2)
  p00 = Psi(Dit(2, 0), Dit(2, 0)).density()

  H_x_I = D.H ^ np.eye(2)
  rho = D.CX | H_x_I | p00

  print(rho)
  for i in range(2):
    for j in range(2):
      prob = Tr(np.array(Psi(Dit(2, i), Dit(2, j)).density().dot(rho)))
      print(f"Proj{i}{j} -> {prob:.2f}")

  return rho

Qubit_Bell()

# # test with d=2,3 to get back paulis and gell-mann matrices
# print("Pauli matrices")
# print(dGellMann(2))
# print("Gell-Mann matrices")
# gm = dGellMann(3)

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



"""
def test_quantum_channel():
    import numpy as np

    # Bit-flip channel with probability p = 0.5
    K0 = np.sqrt(0.5) * np.eye(2)
    K1 = np.sqrt(0.5) * np.array([[0, 1], [1, 0]])
    kraus_ops = [K0, K1]

    # Input state: |0⟩
    rho = np.array([[1, 0], [0, 0]], dtype=complex)

    output = quantum_channel(kraus_ops, rho)
    print("Quantum Channel Output (should be 0.5*|0><0| + 0.5*|1><1|):")
    print(np.round(output, 3))


def test_entanglement_fidelity():
    import numpy as np

    # Depolarizing channel: p = 0.25
    p = 0.25
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    kraus_ops = [
        np.sqrt(1 - 3*p/4) * I,
        np.sqrt(p/4) * X,
        np.sqrt(p/4) * Y,
        np.sqrt(p/4) * Z,
    ]

    # Input state: |0⟩⟨0|
    rho = np.array([[1, 0], [0, 0]], dtype=complex)

    Fe = entanglement_fidelity(rho, kraus_ops)
    print(f"Entanglement Fidelity: {Fe:.4f} (Expected: ~0.8125 for p=0.25)")
    
    
    
    def test_ullmann_fidelity():
    import numpy as np

    psi = np.array([1, 0], dtype=complex)  # |0⟩
    phi = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)  # (|0⟩ + |1⟩)/√2

    f = ullmann_fidelity(psi, phi)
    print(f"Uhlmann Fidelity Test: {f:.4f} (Expected: ~0.7071)")



if __name__ == "__main__":
    test_ullmann_fidelity()
    test_quantum_channel()
    test_entanglement_fidelity()"""
    
'''
def test_negativity():
    d = 3
    psi = np.zeros((d, d), dtype=complex)
    for i in range(d):
        psi[i, i] = 1
    psi = psi.flatten() / np.sqrt(d)
    rho = np.outer(psi, psi.conj())  # Density matrix for maximally entangled qutrit state
    N = negativity(rho, d, d)
    print(f"Qutrit Entangled State Negativity: {N:.4f} (Expected: > 0)")


if __name__ == "__main__":
    test_ullmann_fidelity()
    
    test_negativity()
'''
