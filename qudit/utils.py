from scipy.sparse import kron as skron, csr_matrix
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy import SparseMatrix as Matrix
from .index import Gate, State, VarGate
from typing import Union
from uuid import uuid4
import numpy as np


def ID() -> str:
    return str(uuid4()).split("-")[0]


def isVar(*args) -> bool:
    for arg in args:
        if isinstance(arg, (VarGate, Matrix)):
            return True
    return False


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
def Tensor(*args: Union[Gate, State]):
    if len(args) == 0:
        raise ValueError("At least one arg needed")
    if len(args) == 1:
        return args[0]

    result = args[0]
    for arg in args[1:]:
        result = np.kron(result, arg)

    return result


# # A ^ B ^ C ^ D ^ ... ^ N
def CTensor(*args: Union["Gate", "VarGate", "State"]):
    if not args:
        raise ValueError("At least one arg needed")

    args = list(args)[::-1]
    if isVar(*args):
        result = Matrix(args[0])
        for arg in args[1:]:
            result = TensorProduct(result, Matrix(arg))
    else:
        result = csr_matrix(args[0])
        for arg in args[1:]:
            result = skron(result, csr_matrix(arg))

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
