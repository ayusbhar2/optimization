import cvxopt as cv
import cvxpy as cp
import numpy as np

from solver.classes import AssignmentProblem, BinaryIntegerProblem, TransportationProblem
from solver.utils import is_integer_solution

z_star = -np.inf
var_index = 0
tolerance = 1e-7

def simplex_2D(objective, constraints):
    pass

def dijkstra(G: AdjacencyList, source, destination):
    """Returns shortest path from source to destination in a graph."""
    pass

def transportation_simplex(prob: TransportationProblem):
    pass

def hungarian_method(prob: AssignmentProblem):
    pass

# TODO: write mixed integer version of algo.
def branch_and_bound(bip: BinaryIntegerProblem, var_index=0):
    # Bound
    lp_result = bip.solve_lp_relaxation()
    lp_status = lp_result.get('status')
    lp_value = lp_result.get('optimal_value')
    lp_solution = lp_result.get('optimal_solution')

    result = {'status': None, 'optimal_value': None, 'optimal_solution': None,}

    global z_star

    if (lp_status != 'optimal' or lp_value <= z_star): # fathom: lp infeasible or suboptimal
        result.update(
            {'status': lp_status,
             'optimal_value': -np.inf,})

    elif is_integer_solution(lp_solution, epsilon=tolerance): # fathom: integer solution found
        z_star = max(z_star, lp_value)
        result.update(
            {'status': lp_status,
             'optimal_value': lp_value,
             'optimal_solution': [int(x) for x in lp_solution],})

    elif var_index >= len(bip.variables()): # fathom: reached a leaf node
        result.update(
            {'status': 'optimal',
             'optimal_value': bip.objective.value,
             'optimal_solution': [int(v.value[0]) for v in bip.variables()],})
    else:
        # Branch
        # cvxpy does not guarantee sorting of variables
        split_var = sorted(bip.variables(), key=lambda x: x.id)[var_index]

        # subproblem1
        sub1 = BinaryIntegerProblem(
            bip.objective, bip.constraints + [split_var <= 0])
        result1 = branch_and_bound(sub1, var_index + 1)

        # subproblem2
        sub2 = BinaryIntegerProblem(
            bip.objective, bip.constraints + [split_var >= 1])
        result2 = branch_and_bound(sub2, var_index + 1)

        val1 = result1.get('optimal_value')
        val2 = result2.get('optimal_value')

        z_star = max(z_star, val1, val2)

        if val1 > val2:
            result.update(result1)
        else:
            result.update(result2)

    return result

