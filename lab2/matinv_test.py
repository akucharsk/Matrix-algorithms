import unittest
from scipy.linalg import inv
import matinv as matinv
import numpy as np


def test(ref: unittest.TestCase, size: int):
    matrix = matinv.random_matrix(size)
    my_inv = matinv.invert_matrix(matrix, matinv.binet)
    true_inv = inv(matrix.astype(float))

    ref.assertTrue(np.allclose(my_inv.astype(float), true_inv, rtol=1e-04))


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
