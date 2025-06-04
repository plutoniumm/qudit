from scipy.special import legendre
import numpy as np

class w:
  def __init__(self, d: int):
    self.d = d
    self.value = np.exp(2j * np.pi / d)

  def __str__(self):
    return str(self.value)

  def __repr__(self):
    return str(self.value)

  def __pow__(self, power: int):
    return self.value ** power

"""
Generalised Gell-Mann Matrices
  https://arxiv.org/pdf/0806.1174
  there are d^2 - 1 of them
  (n n-1)/2, (n n-1)/2, n-1
  symmetric, anti symmetric and diagonal (counter)

  Symmetric:
    lambda_(j,k)^s=E_(k,j)+E_(j,k) for 1<=j<k<=n.
  Antisymmetric:
    lambda_(j,k)^a=-i(E_(j,k)-E_(k,j)) for 1<=j<k<=n.
  Diagonal:
    lambda_l=sqrt(2/(l(l+1)))(sum_(j=1)^l
          { E_(j,j)- l.E_(l+1,l+1)) } for 1<=l<=n-1.
"""
def _E(d, j, k):
  mat = np.zeros((d, d))
  mat[j][k] = 1
  return mat

# symmetric
def _L_s(d, j, k):
  return _E(d, j, k) + _E(d, k, j)

# anti-symmetric
def _L_a(d, j, k):
  return 1j * (_E(d, j, k) - _E(d, k, j)).astype(complex)

# diagonal
def _L_d(d, l):
  mat = np.zeros((d, d), dtype=complex)
  for j in range(l):
    mat[j][j] = 1
  mat[l][l] = -l
  return np.sqrt(2 / (l * (l + 1))) * mat

f = 3
# generalised gell mann matrices
def dGellMann(d):
  gell_mann = []
  for j in range(d):
    for k in range(j + 1, d): # k>j
      gell_mann.append(_L_s(d, j, k))
      gell_mann.append(_L_a(d, j, k))

    if j == 0:
      continue
    gell_mann.append(_L_d(d, j))

  return gell_mann

# nth legendre polynomial
def Legendre(y: int, x:int) -> np.ndarray:
  x = np.linspace(-1, 1, x)
  x = legendre(y)(x)
  return np.array(x)/np.linalg.norm(x, 2)

def Orthomat(dim_x:int, dim_y:int) -> np.ndarray:
  x = np.linspace(-1, 1, dim_x)
  mat = [legendre(i)(x) for i in range(dim_y)]

  return np.array(mat)