from typing import Union, List
import torch as pt
import math


def to_mixed(num: int, bases: List[int]) -> List[int]:
    assert len(bases) >= 1, "bases must be non-empty"

    for b in bases:
        assert b >= 2, f"each base must be >= 2, got {b}"

    n = len(bases)
    digits: List[int] = [0] * n

    for k in range(n - 1, -1, -1):
        b = bases[k]
        digits[k] = num % b
        num //= b

    if num != 0:
        raise ValueError

    return digits


def fr_mixed(digits: List[int], bases: List[int]) -> int:
    assert len(digits) == len(bases) and len(bases) >= 1, "digits and bases must have equal non-zero length"
    num = 0

    for d, b in zip(digits, bases):
        if d < 0 or d >= b:
            raise ValueError

        num = num * b + d

    return num


def LittleEndian(
    state: pt.Tensor,
    base: Union[int, List[int], pt.Tensor],
) -> pt.Tensor:
    if not isinstance(state, pt.Tensor):
        state = pt.tensor(state)

    width = state.shape[0]

    if isinstance(base, (list, tuple, pt.Tensor)):
        dims_list = [int(b) for b in base]
    else:
        n = round(math.log(width, base))
        dims_list = [base] * n

    if math.prod(dims_list) != width:
        raise ValueError

    out = pt.empty_like(state)
    dims_rev = dims_list[::-1]

    for i in range(width):
        digits_out = to_mixed(i, dims_rev)
        j = fr_mixed(digits_out[::-1], dims_list)
        out[i] = state[j]

    return out
