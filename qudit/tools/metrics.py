from scipy.linalg import fractional_matrix_power
from typing import List, Union
from .. import Dit, Psi, In
import numpy as np

STATE = Union[Dit, Psi]

def Ullmann(rho, sigma):
  if isinstance(rho, STATE) and isinstance(sigma, STATE):
    return np.absolute(In(rho, sigma))

  if isinstance(rho, STATE):
    rho = rho.density()
  if isinstance(sigma, STATE):
    sigma = sigma.density()

  _2rho = fractional_matrix_power(rho, 0.5)
  inner = np.dot(_2rho, np.dot(sigma, _2rho))

  return np.trace(fractional_matrix_power(inner, 0.5))