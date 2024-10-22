import unittest
import numpy as np
from ai import AI_multiplication
from util import random_matrices_for_ai, Number


def test(ref, k):
    A, B = random_matrices_for_ai(k, 1, 10, np.random.randint)
    ref.assertTrue(np.all(A @ B == AI_multiplication(A, B)))

class MyTestCase(unittest.TestCase):

    def test_4x5X5x5(self):
        test(self, 4)

    def test_4x25X25x25(self):
        test(self, 4)

    def test_4x125X125x125(self):
        test(self, 4)

    def test_16x5X5x5(self):
        test(self, 16)

    def test_16x25X25x25(self):
        test(self, 16)

    def test_64x5X5x5(self):
        test(self, 64)

if __name__ == '__main__':
    unittest.main()