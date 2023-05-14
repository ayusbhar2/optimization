import cvxopt as cv
import cvxpy as cp
import numpy as np

class BinaryIntegerProblem(cp.problems.problem.Problem):

	def __init__(self, objective, constraints):
		for var in objective.variables():
			if not var.attributes['boolean']:
				raise ValueError(
					'All variables must have boolean attribute True for '
					'Binary Integer Programming Problems')

		cp.problems.problem.Problem.__init__(
			self, objective, constraints)

	def solve_lp_relaxation(self):
	    boolean_vars = []
	    new_constraints = []
	    result = {'status': None, 'optimal_value': None, 'optimal_solution': None}

	    for var in self.variables():
	        if var.attributes['boolean']:
	            boolean_vars.append(var)
	            var.attributes['boolean'] = False
	            new_constraints += [0 <= var, var <= 1]
	    lp = cp.Problem(self.objective, self.constraints + new_constraints)
	    lp.solve()

	    result.update(
	        {'status': lp.status,
	         'optimal_value': lp.value})

	    if lp.status == 'optimal':
	        result.update(
	            {'optimal_solution': [v.value[0]for v in lp.variables()]})

	    # reset the boolean state of variables
	    for var in boolean_vars:
	        var.attributes['boolean'] = True

	    return result
