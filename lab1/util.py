import numpy as np


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
    

def random_matrix(k: int, low=1E-8, high=1, function=np.random.uniform) -> np.ndarray:
    return np.array([[Number(j) for j in i] for i in function(low, high, size=(k, k))], dtype=Number)