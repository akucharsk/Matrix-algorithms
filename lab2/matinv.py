import numpy as np
from sys import setrecursionlimit

setrecursionlimit(1000000)

from util import Number, random_matrix, matrix_partitions, number_eye, Pipe
from binet import binet

recursion_counter = 0


def invert_matrix(matrix: np.ndarray, mult) -> np.ndarray:
    if matrix.size == 1:
        return 1 / matrix

    A_11, A_12, A_21, A_22 = matrix_partitions(matrix)
    A_11_inv = invert_matrix(A_11, mult)

    S_22 = A_22 - mult(mult(A_21, A_11_inv), A_12)
    S_22_inv = invert_matrix(S_22, mult)

    long_matmul = mult(mult(mult(A_12, S_22_inv), A_21), A_11_inv)

    B_11 = mult(A_11_inv, long_matmul + number_eye(long_matmul.shape[0]))
    B_12 = -mult(mult(A_11_inv, A_12), S_22_inv)

    B_21 = -mult(mult(S_22_inv, A_21), A_11_inv)

    B_22 = S_22_inv
    N = np.concatenate((B_11, B_12), axis=1)
    S = np.concatenate((B_21, B_22), axis=1)

    return np.concatenate((N, S), axis=0)


if __name__ == '__main__':
    n = 10
    a1 = random_matrix(n, 1, 10, np.random.randint)
    a2 = random_matrix(n, 1, 10, np.random.randint)

    inv = invert_matrix(a1, np.ndarray.__matmul__)
    eye = number_eye(n)
    print(a1, binet(a1, inv), sep='\n\n')
