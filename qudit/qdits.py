from typing import List, Union
import numpy as np

def Out(i: np.ndarray, j: np.ndarray) -> np.ndarray:
  return np.outer(i, j.conj().T)

def In(i: np.ndarray, j: np.ndarray) -> float:
  return np.round(np.dot(i.conj().T, j), 10)
# InO = In(i, O.dot(j)) for <i|O|j>

def UpU(U: np.ndarray, rho: np.ndarray) -> np.ndarray:
  return np.dot(U, np.dot(rho, U.conj().T))

def Tr(A: np.ndarray) -> float:
  return np.real(np.trace(A))

# dit as a wrapper around a numpy array
class Dit(np.ndarray):
  def __new__(cls, d: Union[int, np.ndarray], value: int = None):
    if isinstance(d, int):
      obj = np.zeros(d, dtype=np.complex128).view(cls)
      obj[value] = 1
    else:
      obj = np.asarray(d, dtype=np.complex128).view(cls)
      obj /= np.linalg.norm(obj)

    return obj

  def __array_finalize__(self, obj):
    if obj is None: return

  @property
  def d(self):
    return len(self)

  def density(self) -> np.ndarray:
    return Out(self, self)

  def norm(self) -> 'Dit':
    return Dit(self / np.linalg.norm(self))

  def H(self) -> 'Dit':
    return Dit(self.conj().T)

def Project(*args: List[Dit]) -> np.ndarray:
  state = args[-1]
  for d in args[:-1]:
    state = np.kron(d, state)

  return Out(state, state)

# combined str option and dits option
#  -> Psi(2, "00") or Psi(Dit(2, 0), Dit(2, 1))
class Psi(Dit):
  def __new__(cls, d: Union[int, Dit], *args: List[Union[int, Dit]]):
    if isinstance(d, int):
      string = args[0]
      states = [Dit(d, int(i)) for i in string]
    else:
      states = [d] + list(args)

    state = states[-1]
    for d in states[:-1]:
      state = np.kron(d, state)

    state /= np.linalg.norm(state)
    return state

  def __array_finalize__(self, obj):
    if obj is None: return