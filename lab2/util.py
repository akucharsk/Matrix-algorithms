import numpy as np
from binet import binet


class Number(float):
    add_count = 0
    sub_count = 0
    mul_count = 0
    div_count = 0

    def __new__(cls, value):
        return super().__new__(cls, value)

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
    
    def __radd__(self, other):
        Number.add_count += 1
        return Number(super().__radd__(other))
    
    def __rsub__(self, other):
        Number.sub_count += 1
        return Number(super().__rsub__(other))
    
    def __rmul__ (self, other):
        Number.mul_count += 1
        return Number(super().__rmul__(other))

    def __rtruediv__(self, other):
        Number.div_count += 1
        return Number(super().__rtruediv__(other))

    @classmethod
    def purge(cls):
        cls.add_count = 0
        cls.sub_count = 0
        cls.mul_count = 0
        cls.div_count = 0


class Pipe:
    def __init__(self, value):
        self.value = value

    def chain(self, func, assoc_rvalue=True):
        if assoc_rvalue:
            self.value = func(self.value)
        else:
            func(self.value)
        return self

    def unwrap(self):
        return self.value


def random_matrix(k: int, low=1E-8, high=1, function=np.random.uniform) -> np.ndarray:
    return np.array([[Number(j) for j in i] for i in function(low, high, size=(k, k))], dtype=Number)


type partitioned_matrix = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def matrix_partitions(matrix: np.ndarray) -> partitioned_matrix:
    rows, cols = matrix.shape
    rows //= 2
    cols //= 2

    NW = matrix[:rows, :cols]
    NE = matrix[:rows, cols:]
    SW = matrix[rows:, :cols]
    SE = matrix[rows:, cols:]

    return NW, NE, SW, SE


def number_eye(n: int) -> np.ndarray:
    return np.array([[Number(1 if i == j else 0) for j in range(n)] for i in range(n)], dtype=Number)
