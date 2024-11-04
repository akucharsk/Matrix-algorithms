import unittest
from lu import lu_factorization, random_matrix, binet
import numpy as np


def test(ref: unittest.TestCase, matrix_size: int):
    matrix = random_matrix(matrix_size)
    L, U = lu_factorization(matrix, binet)
    for i in range(matrix_size):
        ref.assertTrue(np.all(L[i, i + 1:] == 0))
        ref.assertTrue(np.all(U[i, :i] == 0))
    ref.assertTrue(np.allclose(L.astype(float) @ U.astype(float), matrix.astype(float)))


class MyTestCase(unittest.TestCase):
    def test_2x2(self):
        test(self, 2)

    def test_4x4(self):
        test(self, 4)

    def test_8x8(self):
        test(self, 8)

    def test_16x16(self):
        test(self, 16)

    def test_32x32(self):
        test(self, 32)

    def test_64x64(self):
        test(self, 64)

    def test_128x128(self):
        test(self, 128)

    def test_256x256(self):
        test(self, 256)


if __name__ == '__main__':
    unittest.main()
