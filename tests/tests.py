import cvxopt as cv
import cvxpy as cp
import numpy as np

import logging
import unittest

from utils import get_lp_relaxation_of_bip


class TestBranchAndBoundForBIP(unittest.TestCase):

    def test_lp_relaxation_of_bip(self):
        # Original BIP (book section 11.6)
        x1 = cp.Variable(1, boolean=True, name='x1')
        x2 = cp.Variable(1, boolean=True, name='x2')
        x3 = cp.Variable(1, boolean=True, name='x3')
        x4 = cp.Variable(1, boolean=True, name='x4')

        obj = cp.Maximize(9 * x1 + 5 * x2 + 6 * x3 + 4 * x4)

        constraints = [6 * x1 + 3 * x2 + 5 * x3 + 2 * x4 <= 10,
                                             x3 +     x4 <= 1,
                          -x1          + x3              <= 0,
                                   -x2              + x4 <= 0]

        bip = cp.Problem(obj, constraints)

        lp = get_lp_relaxation_of_bip(bip)
        lp.solve()

        self.assertAlmostEqual(x1.value[0], 0.8333333)
        self.assertAlmostEqual(x2.value[0], 1)
        self.assertAlmostEqual(x3.value[0], 0)
        self.assertAlmostEqual(x4.value[0], 1)
        self.assertAlmostEqual(lp.value, 16.5)

if __name__ == '__main__':

    unittest.main()
