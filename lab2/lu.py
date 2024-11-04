import numpy as np
from sys import setrecursionlimit
from util import Number, random_matrix, matrix_partitions
from binet import binet
from matinv import invert_matrix


def lu_factorization(matrix: np.ndarray, mult) -> tuple[np.ndarray, np.ndarray]:
    if matrix.size == 1:
        return np.array([[Number(1)]], dtype=Number), matrix

    A_11, A_12, A_21, A_22 = matrix_partitions(matrix)
    L_11, U_11 = lu_factorization(A_11, mult)
    U_11_inv = invert_matrix(U_11, mult)

    L_21 = mult(A_21, U_11_inv)
    L_11_inv = invert_matrix(L_11, mult)

    U_12 = mult(L_11_inv, A_12)
    L_22 = A_22 - mult(mult(mult(A_21, U_11_inv), L_11_inv), A_12)
    S = L_22

    L_S, U_S = lu_factorization(S, mult)
    L_22 = L_S
    U_22 = U_S

    L = np.concatenate(
        (np.concatenate((L_11, np.zeros_like(L_11)), axis=1),
         np.concatenate((L_21, L_22), axis=1)),
        axis=0
    )

    U = np.concatenate(
        (np.concatenate((U_11, U_12), axis=1),
         np.concatenate((np.zeros_like(U_11), U_22), axis=1)),
        axis=0
    )

    return L, U


if __name__ == '__main__':
    n = 4
    A = random_matrix(n, 1, 20)
    L, U = lu_factorization(A, binet)

    print(L, U, A, L @ U, sep='\n\n\n')
