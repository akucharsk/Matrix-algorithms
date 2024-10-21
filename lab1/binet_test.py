import unittest

import numpy as np
from binet import binet, random_matrix


def test(ref, k):
    A, B = random_matrix(k, 0, 10, np.random.randint), random_matrix(k, 0, 10, np.random.randint)
    ref.assertTrue(np.all(A @ B == binet(A, B)))


class MyTestCase(unittest.TestCase):

    def test_1x1(self):
        test(self, 1)

    def test_2x2(self):
        test(self, 2)

    def test_3x3(self):
        test(self, 3)

    def test_5x5(self):
        test(self, 5)

    def test_10x10(self):
        test(self, 10)

    def test_16x16(self):
        test(self, 16)

    def test_32x32(self):
        test(self, 32)

    def test_99x99(self):
        test(self, 99)

    def test_256x256(self):
        test(self, 256)


if __name__ == '__main__':
    unittest.main()
