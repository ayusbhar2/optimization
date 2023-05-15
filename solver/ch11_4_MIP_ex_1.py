import cvxopt as cv
import cvxpy as cp
import numpy as np

M = 100000000

# Variables
y1 = cp.Variable(1, boolean=True)
y2 = cp.Variable(1, boolean=True)
y3 = cp.Variable(1, boolean=True)

z1 = cp.Variable(1, boolean=True)
z2 = cp.Variable(1, boolean=True)

x11 = cp.Variable(1, integer=True)
x21 = cp.Variable(1, integer=True)
x31 = cp.Variable(1, integer=True)

x12 = cp.Variable(1, integer=True)
x22 = cp.Variable(1, integer=True)
x32 = cp.Variable(1, integer=True)

# Constraints
constraints_capacity = [3 * x11 + 4 * x21 + 2 * x31 <= 30,
						4 * x12 + 6 * x22 + 2 * x32 <= 40]
constraints_restriction_1 = [y1 + y2 + y3 <= 2]
constraints_contingency_1 = [x11 + x12 <= M * y1,
							 x21 + x22 <= M * y2,
							 x31 + x32 <= M * y3]
constratins_restriction_2 = [z1 + z2 == 1]	
constraints_contingency_2 = [x11 + x21 + x31 <= M * z1,
							 x12 + x22 + x32 <= M * z2]

constraints_nonneg = [x11 >= 0, x21 >= 0, x31 >= 0,
					  x12 >= 0, x22 >= 0, x32 >= 0]

constraints = constraints_capacity + constraints_restriction_1 + \
	constraints_contingency_1 + constratins_restriction_2 + \
	constraints_contingency_2 + constraints_nonneg

# Objective
obj = cp.Maximize(35 * (x11 + x12) + 35 * (x21 + x22) + 27 * (x31 + x32))

# Problem

prob = cp.Problem(obj, constraints)

prob.solve()

print("status:", prob.status)
print("optimal value", prob.value)
# print("optimal var", x.value, y.value)
