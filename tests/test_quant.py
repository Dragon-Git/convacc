import unittest

import numpy as np

from convacc.quant import requantize


class QuantTest(unittest.TestCase):
    def test_rounding_and_zero_point(self):
        acc = np.array([[[[7, -7, 8, -8]]]], dtype=np.int32)
        bias = np.array([1, 1, 0, 0], dtype=np.int32)
        scale = np.array([3, 3, 1, 1], dtype=np.int32)
        shift = np.array([1, 1, 1, 1], dtype=np.uint8)
        zp = np.array([0, 0, -2, 2], dtype=np.int8)
        actual = requantize(acc, bias, scale, shift, zp)
        np.testing.assert_array_equal(actual, np.array([[[[12, -9, 2, -2]]]], dtype=np.int8))

    def test_saturation(self):
        acc = np.array([[200, -200]], dtype=np.int32)
        actual = requantize(acc, np.zeros(2, np.int32), np.ones(2, np.int32), np.zeros(2, np.uint8), np.zeros(2, np.int8))
        np.testing.assert_array_equal(actual, np.array([[127, -128]], dtype=np.int8))


if __name__ == "__main__":
    unittest.main()
