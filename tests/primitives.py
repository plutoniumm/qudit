import sys

sys.path.append("..")
from qudit import *

np.set_printoptions(precision=4, suppress=True)


def basis():
    Ket = Basis(2)
    print("00: ", Ket(0, 0), Ket("00"))
    print("11: ", Ket(1, 1), Ket("11"))


def sv():
    pi = np.pi
    e, rt = np.exp, np.sqrt
    cos, sin = np.cos, np.sin

    w = Unity(3)
    Ket = Basis(4)
    SV = State(
        w * Ket("0000")
        + w**2 * Ket("1010")
        + rt(3) * 1j * Ket("2010")
        + Ket("2200")
        + (9j + 16) * Ket("1210")
        + (w - w**2) * Ket("0022")
        + (w - 1) ** 2 * Ket("2020")
        + (e(1j * pi / 18) + 6) * Ket("2221")
        + Ket("0112")
        + (5 + 9j) * Ket("1200")
        + 0.67 * Ket("1111")
        + (9 * cos(pi / 16) + 1j * sin(pi / 5)) * Ket("2222")
    )

    return SV


if __name__ == "__main__":
    basis()
    print(sv())
