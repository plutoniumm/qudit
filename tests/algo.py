import sys

sys.path.append("..")


# from sympy import exp, SparseMatrix, Symbol
from unittest import TestCase, main
from qudit.algo import Statiliser
import numpy as np


statiliser = Statiliser(["ZZZII", "IIZZZ", "XIXXI", "IXXIX"])

states = statiliser.generate()
print(states.round(3))