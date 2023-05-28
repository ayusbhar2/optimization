import cvxopt as cv
import cvxpy as cp
import numpy as np

from solver.classes import AssignmentProblem, BinaryIntegerProblem, Graph
from solver.classes import TransportationProblem
from solver.utils import is_integer_solution

z_star = -np.inf
var_index = 0
tolerance = 1e-7

def _extract_path(previous_nodes, target_node, path=''):
    path = target_node + path
    if not previous_nodes[target_node]:
        return path
    else:
        u = previous_nodes[target_node]
        return _extract_path(previous_nodes, u, path=path)

def simplex_2D(objective, constraints):
    pass

def get_shortest_paths(graph: Graph, source: str, target=None, algorithm='dijkstra'):
    """Returns shortest paths from source to every node in a graph."""

    dists = {source: 0} # maps each node to its shortest distance from source
    previous_nodes = {source: None} # node preceding the current node in shortest path

    while len(dists) < len(graph):
        candidates = []
        for edge, cost in graph.edge_costs.items():
            if (edge[0] in dists and edge[1] not in dists): # edge crosses frontier
                candidates.append((edge, dists[edge[0]] + cost))
            else:
                continue

        sorted_candidates = sorted(candidates, key=lambda x: x[1]) # sort by cost

        n = len(sorted_candidates)
        if n == 0:
            raise ValueError('No more edges to explore!')
        else:
            best_candidates = [sorted_candidates[0]]
            j = 0
            while (j + 1 < n and sorted_candidates[j][1] == sorted_candidates[j + 1][1]):
                best_candidates.append(sorted_candidates[j + 1])
                j += 1
        for edge, dist in best_candidates:
            dists.update({edge[1]: dist})
            previous_nodes.update({edge[1]: edge[0]})

    if target: # a target node was provided
        shortest_path = _extract_path(previous_nodes, target)
        shortest_distance = dists[target]
        return shortest_distance, shortest_path

    else:
        return dists, previous_nodes

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

