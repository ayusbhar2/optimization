import cvxopt as cv
import cvxpy as cp
import numpy as np

from abc import ABC, abstractmethod


class Graph(dict):
    # TODO: add check for appropriate dict structure
    def __init__(self, edges_to_costs_map: dict):
        dict.__init__(self, edges_to_costs_map)

        self.vertices = set()
        if self.items(): # non-empty graph
            for edge, cost in self.items():
                self.vertices.add(edge[0])
                self.vertices.add(edge[1])

class NetworkProblem(ABC):
    @abstractmethod
    def solve(self):
        pass

class TransportationProblem(NetworkProblem):
    def solve():
        pass

class AssignmentProblem(NetworkProblem):
    def solve():
        pass

class MaxFlowProblem(NetworkProblem):
    def solve():
        pass

class IntegerProblem(cp.problems.problem.Problem):

    def __init__(self, objective, constraints):
        # TODO: check that at least one of the variables is integer
        cp.problems.problem.Problem.__init__(
            self, objective, constraints)

class MixedIntegerProblem(IntegerProblem):

    def __init__(self, objective, constraints):
        IntegerProblem.__init__(
            self, objective, constraints)

class BinaryIntegerProblem(IntegerProblem):

    def __init__(self, objective, constraints):
        for var in objective.variables():
            if not var.attributes['boolean']:
                raise ValueError(
                    'All variables must have boolean attribute True for '
                    'Binary Integer Programming Problems')

        IntegerProblem.__init__(
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
