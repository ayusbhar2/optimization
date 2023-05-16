import cvxpy as cp
import cvxopt as cv
import numpy as np

from solver.utils import get_result_summary

# 1. Linear economic dispatch model (no setup costs)

p1 = cp.Variable(1, nonneg=True, name='p1')
p2 = cp.Variable(1, nonneg=True, name='p2')
p3 = cp.Variable(1, nonneg=True, name='p3')

obj1 = cp.Minimize(7 * p1 + 8 * p2 + 9 * p3)

constraints1 = [p1 + p2 + p3 >= 700,
                p1 >= 250,
                p1 <= 600,
                p2 >= 200,
                p2 <= 400,
                p3 >= 150,
                p3 <= 500]

prob1 = cp.Problem(obj1, constraints1)
prob1.solve()

summary = get_result_summary(prob1)

print(summary)


# 2. Model including setup costs

M = 1000000
u1 = cp.Variable(1, boolean=True, name='u1')
u2 = cp.Variable(1, boolean=True, name='u2')
u3 = cp.Variable(1, boolean=True, name='u3')

obj2 = cp.Minimize(510 * u1 + 7 * p1 + 
                   310 * u2 + 8 * p2 + 
                   78 * u3  + 9 * p3)

constraints2 = [p1 + p2 + p3 >= 700,
                p1 >= 250 - M * (1 - u1),
                p1 <= 600 + M * (1 - u1),
                p2 >= 200 - M * (1 - u2),
                p2 <= 400 + M * (1 - u2),
                p3 >= 150 - M * (1 - u3),
                p3 <= 500 + M * (1 - u3),
                p1 <= M * u1,
                p2 <= M * u2,
                p3 <= M * u3,]

prob2 = cp.Problem(obj2, constraints2)
prob2.solve()

summary = get_result_summary(prob2)
print(summary)

# 3. Model with demand profile

p11 = cp.Variable(1, nonneg=True, name='p11')
p21 = cp.Variable(1, nonneg=True, name='p21')
p31 = cp.Variable(1, nonneg=True, name='p31')

p12 = cp.Variable(1, nonneg=True, name='p12')
p22 = cp.Variable(1, nonneg=True, name='p22')
p32 = cp.Variable(1, nonneg=True, name='p32')

p13 = cp.Variable(1, nonneg=True, name='p13')
p23 = cp.Variable(1, nonneg=True, name='p23')
p33 = cp.Variable(1, nonneg=True, name='p33')

u11 = cp.Variable(1, boolean=True, name='u11')
u21 = cp.Variable(1, boolean=True, name='u21')
u31 = cp.Variable(1, boolean=True, name='u31')

u12 = cp.Variable(1, boolean=True, name='u12')
u22 = cp.Variable(1, boolean=True, name='u22')
u32 = cp.Variable(1, boolean=True, name='u32')

u13 = cp.Variable(1, boolean=True, name='u13')
u23 = cp.Variable(1, boolean=True, name='u23')
u33 = cp.Variable(1, boolean=True, name='u33')

obj3 = cp.Minimize(
    510 * (u11 + u12 + u13) + 7 * (p11 + p12 + p13) +
    310 * (u21 + u22 + u23) + 8 * (p21 + p22 + p23) +
    78  * (u31 + u32 + u33) + 9 * (p31 + p32 + p33))

constraints3 = [p11 + p21 + p31 >= 600,
                p12 + p22 + p32 >= 800,
                p13 + p23 + p33 >= 700,

                p11 >= 250 - M * (1 - u11),
                p11 <= 600 + M * (1 - u11),
                p12 >= 250 - M * (1 - u12),
                p12 <= 600 + M * (1 - u12),
                p13 >= 250 - M * (1 - u13),
                p13 <= 600 + M * (1 - u13),

                p21 >= 200 - M * (1 - u21),
                p21 <= 400 + M * (1 - u21),
                p22 >= 200 - M * (1 - u22),
                p22 <= 400 + M * (1 - u22),
                p23 >= 200 - M * (1 - u23),
                p23 <= 400 + M * (1 - u23),

                p31 >= 150 - M * (1 - u31),
                p31 <= 500 + M * (1 - u31),
                p32 >= 150 - M * (1 - u32),
                p32 <= 500 + M * (1 - u32),
                p33 >= 150 - M * (1 - u33),
                p33 <= 500 + M * (1 - u33),

                p11 <= M * u11,
                p21 <= M * u21,
                p31 <= M * u31,

                p12 <= M * u12,
                p22 <= M * u22,
                p32 <= M * u32,

                p13 <= M * u13,
                p23 <= M * u23,
                p33 <= M * u33]

prob3 = cp.Problem(obj3, constraints3)
prob3.solve()
summary = get_result_summary(prob3)
print(summary)


