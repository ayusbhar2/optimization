import cvxopt as cv
import cvxpy as cp
import numpy as np

import logging
import unittest

from algorithms import branch_and_bound
from problems import BinaryIntegerProblem
from utils import is_integer_solution


class TestBinaryIntegerProblem(unittest.TestCase):

    def test_lp_relaxation(self):
        # Original BIP (book section 11.6)
        x1 = cp.Variable(1, boolean=True, name='x1')
        x2 = cp.Variable(1, boolean=True, name='x2')
        x3 = cp.Variable(1, boolean=True, name='x3')
        x4 = cp.Variable(1, boolean=True, name='x4')

        obj = cp.Maximize(9 * x1 + 5 * x2 + 6 * x3 + 4 * x4)

        constraints = [6 * x1 + 3 * x2 + 5 * x3 + 2 * x4 <= 10,
                                             x3 +     x4 <= 1,
                          -x1              + x3          <= 0,
                                   -x2              + x4 <= 0]

        bip = BinaryIntegerProblem(obj, constraints)
        result = bip.solve_lp_relaxation()

        self.assertAlmostEqual(x1.value[0], 0.8333333)
        self.assertAlmostEqual(x2.value[0], 1)
        self.assertAlmostEqual(x3.value[0], 0)
        self.assertAlmostEqual(x4.value[0], 1)

        self.assertAlmostEqual(result['optimal_value'], 16.5)

        self.assertTrue(x1.attributes['boolean'])
        self.assertTrue(x2.attributes['boolean'])
        self.assertTrue(x3.attributes['boolean'])
        self.assertTrue(x4.attributes['boolean'])

    def test_boolean_false_raises_exception(self):
        # Original BIP (book section 11.6)
        x1 = cp.Variable(1, boolean=False, name='x1')
        obj = cp.Maximize(x1 ** 2)
        constraints = [x1 <= 1]

        with self.assertRaises(ValueError):
            bip = BinaryIntegerProblem(obj, constraints)

class TestBranchAndBound(unittest.TestCase):

    def test_branch_and_bound(self):
        # Original BIP (book section 11.6)
        x1 = cp.Variable(1, boolean=True, name='x1')
        x2 = cp.Variable(1, boolean=True, name='x2')
        x3 = cp.Variable(1, boolean=True, name='x3')
        x4 = cp.Variable(1, boolean=True, name='x4')

        obj = cp.Maximize(9 * x1 + 5 * x2 + 6 * x3 + 4 * x4)

        constraints = [6 * x1 + 3 * x2 + 5 * x3 + 2 * x4 <= 10,
                                             x3 +     x4 <= 1,
                          -x1              + x3          <= 0,
                                   -x2              + x4 <= 0]

        bip = BinaryIntegerProblem(obj, constraints)
        result = branch_and_bound(bip)

        self.assertEqual(result.get('status'), 'optimal')
        self.assertAlmostEqual(result.get('optimal_value'), 14)
        self.assertEqual(result.get('optimal_solution')[0], 1)
        self.assertEqual(result.get('optimal_solution')[1], 1)
        self.assertEqual(result.get('optimal_solution')[2], 0)
        self.assertEqual(result.get('optimal_solution')[3], 0)

class TestUtils(unittest.TestCase):

    def test_is_integer_solution(self):
        self.assertTrue(is_integer_solution([1e-7], 1e-7))
        self.assertFalse(is_integer_solution([1.5], 1e-7))


if __name__ == '__main__':

    unittest.main()
