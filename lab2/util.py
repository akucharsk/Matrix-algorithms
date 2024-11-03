import numpy as np


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

def random_matrix(k: int, low=1E-8, high=1, function=np.random.uniform) -> np.ndarray:
    return np.array([[Number(j) for j in i] for i in function(low, high, size=(k, k))], dtype=Number)



def random_matrices_for_ai(n: int, m: int, low=1E-8, high=1, function=np.random.uniform) -> tuple[np.ndarray, np.ndarray]:
    #n = m = k
    # pow_of_4 = [4 ** i for i in range(1, 10)]
    # pow_of_5 = [5 ** i for i in range(1, 10)]

    A = np.array([[Number(j) for j in i] for i in np.random.uniform(low=10**(-8), high=1, size = (n, m))], dtype=Number)
    # B = np.array([[Number(val) for val in row] for row in B])
    B = np.array([[Number(j) for j in i] for i in np.random.uniform(low=10**(-8), high=1, size = (m, m))], dtype=Number)

    return A, B
