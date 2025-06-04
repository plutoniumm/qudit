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