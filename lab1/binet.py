import numpy as np
from sys import setrecursionlimit
from util import Number
setrecursionlimit(100000)


type block_matrix = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def binet_partition(matrix: np.ndarray) -> block_matrix:
    n = matrix.shape[0] // 2
    m = matrix.shape[1] // 2
    
    NW = matrix[:n, :m]
    NE = matrix[:n, m:]
    SW = matrix[n:, :m]
    SE = matrix[n:, m:]

    return NW, NE, SW, SE


def binet(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.shape[0] <= 2 or B.shape[0] <= 2:

        return A @ B

    A_11, A_12, A_21, A_22 = binet_partition(A)

    B_11, B_12, B_21, B_22 = binet_partition(B)

    NW = binet(A_11, B_11) + binet(A_12, B_21)
    NE = binet(A_11, B_12) + binet(A_12, B_22)
    SW = binet(A_21, B_11) + binet(A_22, B_21)
    SE = binet(A_21, B_12) + binet(A_22, B_22)

    N = np.concatenate((NW, NE), axis=1)
    S = np.concatenate((SW, SE), axis=1)
    return np.concatenate((N, S))




def generate_tests() -> str:
    result = ""
    for i in [1, 2, 3, 5, 10, 16, 32, 99, 256]:
        result += f"def test_{i}x{i}(self):\ntest(self, {i})\n\n"
    return result


if __name__ == '__main__':
    print(generate_tests())
