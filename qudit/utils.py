from .index import State
from typing import Union
import torch as pt


# <A|b@c@d@e...@n|B>
def Braket(*args: pt.Tensor) -> pt.Tensor:
    if len(args) < 2:
        raise ValueError("At least two arguments are required for Braket")

    args = list(args)
    args = [pt.as_tensor(a, dtype=pt.complex128) for a in args]

    args[-1] = args[-1].conj().T
    result = args[0]
    for arg in args[1:]:
        result = result @ arg

    return result


# # A ^ B ^ C ^ D ^ ... ^ N
def Tensor(*args: Union[pt.Tensor, State]):
    if len(args) == 0:
        raise ValueError("At least one arg needed")
    if len(args) == 1:
        return args[0]

    result = pt.as_tensor(args[0], dtype=pt.complex128)
    for arg in args[1:]:
        result = pt.kron(result, pt.as_tensor(arg, dtype=pt.complex128))

    return result


class partial:
    @staticmethod
    def trace(rho: pt.Tensor, dA: int, dB: int, keep: str = "A") -> pt.Tensor:

        assert rho.shape == (
            dA * dB,
            dA * dB,
        ), "Input must be a square matrix of shape (dA*dB, dA*dB)"
        rho = pt.as_tensor(rho, dtype=pt.complex128).reshape(dA, dB, dA, dB)

        if keep == "A":
            return pt.einsum("abcb->ac", rho)  # Result: shape (dA, dA)
        elif keep == "B":
            return pt.einsum("abac->bc", rho)  # Result: shape (dB, dB)
        else:
            raise ValueError("keep must be 'A' or 'B'")

    @staticmethod
    def transpose(rho: pt.Tensor, dim_A: int, dim_B: int) -> pt.Tensor:
        assert rho.shape == (
            dim_A * dim_B,
            dim_A * dim_B,
        ), "Input must be a square matrix of shape (dim_A*dim_B, dim_A*dim_B)"

        rho = pt.as_tensor(rho, dtype=pt.complex128).reshape(dim_A, dim_B, dim_A, dim_B)
        rho_pt = rho.permute(0, 3, 2, 1)

        return rho_pt.reshape(dim_A * dim_B, dim_A * dim_B)
