from lu import lu_factorization
from binet import binet
import numpy as np

def recursive_det(A, mult):
    L, U = lu_factorization(A, mult)
    return U.diagonal().prod()

