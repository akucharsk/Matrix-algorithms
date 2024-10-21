import numpy as np
from sys import setrecursionlimit

setrecursionlimit(100000)


class Number(float):
    add_count = 0
    sub_count = 0
    mul_count = 0
    div_count = 0

    def __add__(self, other):
        Number.add_count += 1
        return Number(super().__add__(other))

    def __sub__(self, other):
        Number.sub_count += 1
        return Number(super().__sub__(other))

    def __mul__(self, other):
        Number.mul_count += 1
        return Number(super().__mul__(other))

    def __truediv__(self, other):
        Number.div_count += 1
        return Number(super().__truediv__(other))


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


def random_matrix(k: int, low=1E-8, high=1, function=np.random.uniform) -> np.ndarray:
    return np.array([[Number(j) for j in i] for i in function(low, high, size=(k, k))], dtype=Number)


def generate_tests() -> str:
    result = ""
    for i in [1, 2, 3, 5, 10, 16, 32, 99, 256]:
        result += f"def test_{i}x{i}(self):\ntest(self, {i})\n\n"
    return result


if __name__ == '__main__':
    print(generate_tests())
