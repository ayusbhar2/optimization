import cvxopt as cv
import cvxpy as cp
import numpy as np

from abc import ABC, abstractmethod

class Edge():
    def __init__(self, source, target, **kwargs):
        self.source = source
        self.target = target
        for k, v in kwargs.items():
            self.__dict__[k] = v

    def __eq__(self, other):
        if (self.source == other.source and self.target == other.target):
            return True
        else:
            return False

    def __str__(self):
        return 'Edge: {}->{}'.format(self.source, self.target)

    def __repr__(self):
        return 'Edge: {}->{}'.format(self.source, self.target)

    def __hash__(self):
        return hash(repr(self))

class Graph():
    def __init__(self, edge_list=[]):
        self.edges = set()
        self.vertices = set()
        if edge_list:
            for edge in edge_list:
                self.add_edge(edge)
                self.vertices.add(edge.source)
                self.vertices.add(edge.target)

    def add_edge(self, edge: Edge):
        if edge in self.edges:
            raise ValueError('Cannot add {}. Edge alrady exists.'.format(edge))
        self.edges.add(edge)
        self.vertices.add(edge.source)
        self.vertices.add(edge.target)

    def get_edge(self, source, target):
        for edge in self.edges:
            if edge.source == source and edge.target == target:
                return edge
        raise KeyError('The requested edge does not exist!')

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
