import cvxopt as cv
import cvxpy as cp
import numpy as np

from algorithms import branch_and_bound
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

# cvxpy
bip.solve()
print('status: ', bip.status)
print('optimal value: ', bip.value)
print('optimal solution: ', [v.value[0] for v in bip.variables()])

# my naive implementation
result = branch_and_bound(bip)
print('status: ', result.get('status'))
print('optimal value: ', result.get('optimal_value'))
print('optimal solution: ', result.get('optimal_solution'))
