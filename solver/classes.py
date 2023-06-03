import cvxpy as cp

from abc import ABC, abstractmethod


class Vertex():
    def __init__(self, name: str, **kwargs):
        self.name = name
        for k, v in kwargs.items():
            self.__dict__[k] = v

    def __eq__(self, other):
        if self.name == other.name:
            return True
        else:
            return False

    def __str__(self):
        return 'Vertex: {}'.format(self.name)

    def __repr__(self):
        return 'Vertex: {}'.format(self.name) 

    def __hash__(self):
        return hash(repr(self))

    def add_attr(self, **kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = v


class Edge():
    def __init__(self, source, target, **kwargs):
        if not isinstance(source, Vertex):
            source = Vertex(name=source)

        if not isinstance(target, Vertex):
            target = Vertex(name=target)

        self.source = source
        self.target = target
        for k, v in kwargs.items():
            self.__dict__[k] = v

    def __eq__(self, other):
        if (self.source == other.source and
                self.target == other.target):
            return True
        else:
            return False

    def __str__(self):
        return 'Edge: {}->{}'.format(self.source.name, self.target.name)

    def __repr__(self):
        return 'Edge: {}->{}'.format(self.source.name, self.target.name)

    def __hash__(self):
        return hash(repr(self))

    def update(self, **kwargs):
        self.__dict__.update(kwargs)

    def delete_attr(self, name: str):
        self.__dict__.pop(name)


class Graph():
    def __init__(self, edge_list=[]):
        self.edges = set()
        self.vertices = set()
        if edge_list:
            for edge in edge_list:
                self.add_edge(edge)

    def add_edge(self, edge: Edge):
        if edge in self.edges:
            raise ValueError('Cannot add {}. Edge alrady exists.'.format(edge))
        self.edges.add(edge)
        self.vertices.add(edge.source)
        self.vertices.add(edge.target)

    def get_edge(self, source_name, target_name):
        for edge in self.edges:
            if (edge.source.name == source_name and
                    edge.target.name == target_name):
                return edge
        raise KeyError('The requested edge does not exist!')

    def get_edges(self, source_name=None, target_name=None):
        result = []
        if not source_name and not target_name:
            return self.edges
        if source_name and not target_name:
            for edge in self.edges:
                if edge.source.name == source_name:
                    result.append(edge)
            return result
        if target_name and not source_name:
            for edge in self.edges:
                if edge.target.name == target_name:
                    result.append(edge)
            return result
        result.append(self.get_edge(source_name, target_name))
        return result

    def update_edge(self, source_name, target_name, **kwargs):
        for edge in self.edges:
            if (edge.source.name == source_name and
                    edge.target.name == target_name):
                edge.update(**kwargs)
                return
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
        result = {'status': None,
                  'optimal_value': None,
                  'optimal_solution': None}

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
