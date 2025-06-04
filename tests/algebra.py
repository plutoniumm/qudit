import sys

sys.path.append("..")
from qudit import *


def gellMann():
    gm = dGellMann(3)
    for mat in gm:
        print(mat)
    print("=" * 10, "\nNumber of Gell-Mann matrices:\n\t", len(gm))


if __name__ == "__main__":
    gellMann()
