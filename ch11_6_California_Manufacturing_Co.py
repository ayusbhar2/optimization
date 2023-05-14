import cvxopt as cv
import cvxpy as cp
import numpy as np

from problems import BinaryIntegerProblem

# Original BIP
x1 = cp.Variable(1, boolean=True, name='x1')
x2 = cp.Variable(1, boolean=True, name='x2')
x3 = cp.Variable(1, boolean=True, name='x3')
x4 = cp.Variable(1, boolean=True, name='x4')

obj = cp.Maximize(9 * x1 + 5 * x2 + 6 * x3 + 4 * x4)

constraints = [6 * x1 + 3 * x2 + 5 * x3 + 2 * x4 <= 10,
                                     x3 +     x4 <= 1,
                  -x1          + x3              <= 0,
                           -x2              + x4 <= 0]

bip = BinaryIntegerProblem(obj, constraints)

# ~ Branch ~ #

# Subprob 1 (x1 = 0)
sub1 = BinaryIntegerProblem(obj, constraints + [x1 <= 0])
result1 = sub1.solve_lp_relaxation()
