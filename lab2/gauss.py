import numpy as np
from lu import lu_factorization
from util import Number, random_matrix, matrix_partitions
from matinv import invert_matrix
from binet import binet
from strassen import Strassen, SMU

def gauss_elimination(A: np.ndarray,b: np.ndarray, mult) -> np.ndarray:

    if A.shape[0] == 1:
        return b / A[0, 0]

    A11, A12,A21, A22 = matrix_partitions(A)
    n = A.shape[0]//2


    # b1 = b[:n]
    # b2 = b[n:]
    b1 = b[:n].reshape(-1, 1)
    b2 = b[n:].reshape(-1, 1)

    
    L11, U11 = lu_factorization(A11, mult=mult)

    L11_inv = invert_matrix(L11, mult)
    U11_inv = invert_matrix(U11, mult=mult)
    S = A22 - mult(mult(mult(A21, U11_inv), L11_inv), A12)

    L_S, U_S = lu_factorization(S, mult=mult)

    C11 = U11
    C12 = mult(L11_inv, A12)

    C22 = U_S

    RHS1 = mult(L11_inv, b1)
    RHS2 = mult(invert_matrix(L_S, mult=mult), b2 - mult(mult(mult(A21, U11_inv), L11_inv), b1))

    x2 = mult(invert_matrix(U_S, mult=mult), RHS2)

    x1 = mult(invert_matrix(U11, mult=mult), RHS1 - mult(C12, x2))

    x = np.concatenate((x1, x2)).flatten()


    return x
              


     



if __name__ == '__main__':
    np.random.seed(42)  # For reproducibility

    # Generate a random A matrix and b vector
    n = 2  # Size of the matrix
    A = np.random.rand(n, n)
    b = np.random.rand(n)

    print(A, b)

    # Solve using recursive Gaussian elimination
    x_recursive = gauss_elimination(A, b, SMU)
    print(x_recursive)

    # Solve using numpy's built-in function
    x_numpy = np.linalg.solve(A, b)

    if np.allclose(x_recursive, x_numpy):
        print("\nThe solutions match! The implementation is correct.")
    else:
        print("\nThe solutions do not match. There may be an error in the implementation.")

