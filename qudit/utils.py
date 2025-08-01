from .index import State
from typing import Union
import numpy as np


# <A|b@c@d@e...@n|B>
def Braket(*args: np.ndarray) -> np.ndarray:
    if len(args) < 2:
        raise ValueError("At least two arguments are required for Braket")

    args = list(args)
    args[-1] = args[-1].conj().T
    result = args[0]
    for arg in args[1:]:
        result = np.dot(result, arg)

    return result


# # A ^ B ^ C ^ D ^ ... ^ N
def Tensor(*args: Union[np.ndarray, State]):
    if len(args) == 0:
        raise ValueError("At least one arg needed")
    if len(args) == 1:
        return args[0]

    result = args[0]
    for arg in args[1:]:
        result = np.kron(result, arg)

    return result


class partial:
    @staticmethod
    def trace(rho: np.ndarray, dA: int, dB: int, keep: str = "A") -> np.ndarray:

        assert rho.shape == (
            dA * dB,
            dA * dB,
        ), "Input must be a square matrix of shape (dA*dB, dA*dB)"
        rho = rho.reshape(dA, dB, dA, dB)

        if keep == "A":
            return np.trace(rho, axis1=1, axis2=3)  # Result: shape (dA, dA)
        elif keep == "B":
            return np.trace(rho, axis1=0, axis2=2)  # Result: shape (dB, dB)
        else:
            raise ValueError("keep must be 'A' or 'B'")

    @staticmethod
    def transpose(rho: np.ndarray, dim_A: int, dim_B: int) -> np.ndarray:
        assert rho.shape == (
            dim_A * dim_B,
            dim_A * dim_B,
        ), "Input must be a square matrix of shape (dim_A*dim_B, dim_A*dim_B)"

        rho = rho.reshape(dim_A, dim_B, dim_A, dim_B)
        rho_pt = np.transpose(rho, axes=(0, 3, 2, 1))
        return rho_pt.reshape(dim_A * dim_B, dim_A * dim_B)
