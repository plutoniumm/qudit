from .algebra import w, dGellMann
from typing import List, Union
from .qdits import Dit
import numpy as np
import math as ma

# gate as a wrapper around a numpy.ndarray
class Gate(np.ndarray):
  def __new__(cls, d: int, O: np.ndarray=None, name: str = None):
    if O is None:
      obj = np.zeros((d, d), dtype=complex).view(cls)
      obj.sz = 1
    else:
      obj = np.asarray(O, dtype=complex).view(cls)
      obj.sz = ma.log(len(O[0]), d)
    # endif

    obj.name = name if name else str(d)
    obj.d = d

    return obj

  @property
  def H(self):
    return self.conj().T

  def __array_finalize__(self, obj):
    if obj is None: return
    self.d = getattr(obj, 'd', None)
    self.sz = getattr(obj, 'sz', None)
    self.name = getattr(obj, 'name', None)

  def is_unitary(self):
    return np.allclose(self @ self.H, np.eye(self.shape[0]))

  def is_hermitian(self):
    return np.allclose(self, self.H)

  # gate1 ^ gate2 for np.kron(gate1, gate2)
  def __xor__(self, other):
    if not isinstance(other, Gate):
      other = Gate(other.shape[0], other)

    return Gate(self.d * other.d, np.kron(other, self), f"{self.name}^{other.name}")

  # gate1 | Dit for np.dot(gate1, Dit)
  def __or__(self, other: Union[Dit, 'Gate']):
    if isinstance(other, Dit):
      return np.dot(self, other)
    else:
      return np.dot(self, np.dot(other, self.H))

ck = 23
# special class to create "d" once and pass through all gates
# so G = DGate(d) -> G.X -> G.Z -> G.H -> ...
class DGate:
  def __init__(self, d: int):
    self.d = d

  @property
  def X(self):
    O = np.zeros((self.d, self.d))
    O[0, self.d - 1] = 1
    O[1:, 0:self.d - 1] = np.eye(self.d - 1)
    return Gate(self.d, O, "X")

  @property
  def CX(self):
    perm = self.X

    # Sum of X^k ⊗ |k><k|
    O = sum(
      np.kron(
        np.linalg.matrix_power(perm, k),
        Dit(self.d, k).density()
      ) for k in range(self.d)
    )

    return Gate(self.d**2, O, "CX")

  @property
  def Z(self):
    O = np.diag([w(self.d)**i for i in range(self.d)])
    return Gate(self.d, O, "Z")

  @property
  def H(self):
    O = np.zeros((self.d, self.d), dtype=complex)
    for j in range(self.d):
      for k in range(self.d):
        O[j, k] = w(self.d)**(j*k) / np.sqrt(self.d)

    return Gate(self.d, O, "H")

  def Rot(self, thetas: List[complex]):
    R = np.eye(self.d)
    for i, theta in enumerate(thetas):
      R = np.exp(-1j * theta * dGellMann(self.d)[i]) @ R

    return Gate(self.d, R, "Rot")

  @property
  def I(self):
    return Gate(self.d, np.eye(self.d), "I")