import unittest

import numpy as np

from util import random_matrix
from det import recursive_det
from binet import binet

def test(ref, size):
    mat = random_matrix(size)
    d = recursive_det(mat, binet)
    ref.assertTrue(np.allclose(d, np.linalg.det(mat.astype(float))))

class MyTestCase(unittest.TestCase):
    def test_2x2(self):
        test(self, 2)

    def test_14x14(self):
        test(self, 14)

    def test_26x26(self):
        test(self, 26)

    def test_38x38(self):
        test(self, 38)

    def test_50x50(self):
        test(self, 50)

    def test_62x62(self):
        test(self, 62)

    def test_74x74(self):
        test(self, 74)

    def test_86x86(self):
        test(self, 86)

    def test_98x98(self):
        test(self, 98)

    def test_110x110(self):
        test(self, 110)

    def test_122x122(self):
        test(self, 122)

    def test_134x134(self):
        test(self, 134)

    def test_146x146(self):
        test(self, 146)

if __name__ == '__main__':
    unittest.main()
