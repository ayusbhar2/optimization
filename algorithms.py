import cvxopt as cv
import cvxpy as cp
import numpy as np

from problems import BinaryIntegerProblem
from utils import is_integer_solution

z_star = -np.inf
tolerance = 0.00000001

def branch_and_bound(bip: BinaryIntegerProblem):
    result = bip.solve_lp_relaxation()
    lp_status = result.get('status')
    lp_value = result.get('optimal_value')
    lp_solution = result.get('optimal_solution')

    global z_star

    if (lp_status != 'optimal' or lp_value <= z_star): # fathom: lp infeasible
        return (-np.inf, None)

    elif is_integer_solution(lp_solution, epsilon=tolerance): # fathom: integer solution found
        z_star = max(z_star, lp_value)
        return (lp_value, lp_solution)

    else:
        pass
