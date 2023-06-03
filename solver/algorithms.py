import numpy as np

import logging

from solver.classes import AssignmentProblem, BinaryIntegerProblem, Graph
from solver.classes import TransportationProblem
from solver.utils import is_integer_solution

z_star = -np.inf
var_index = 0
tolerance = 1e-7

logging.basicConfig(level=logging.ERROR)


def _extract_path(previous_nodes, target_node, path=[]):
    path = [target_node] + path
    if not previous_nodes[target_node]:
        return path
    else:
        u = previous_nodes[target_node]
        return _extract_path(previous_nodes, u, path=path)


def simplex_2D(objective, constraints):
    pass


def get_shortest_path(graph: Graph, source_name: str, target_name='',
                      algorithm='dijkstra'):
    """Get shortest path from source to target in connected, undirected graph.
    """
    logging.info(
        'Starting the shortest path algorithm with source: {}'.format(
            source_name))
    try:
        dists = {source_name: 0}  # holds shortest distances from source node
        previous_nodes = {source_name: None}  # stores preceding node

        i = 1
        while len(dists) < len(graph.vertices):
            logging.info('Iteration {}'.format(i))

            candidates = []
            for edge in graph.edges:
                if (edge.source.name in dists and
                        edge.target.name not in dists):  # edge crosses frontier
                    dist = dists[edge.source.name] + edge.cost
                    candidates.append((edge, dist))
                else:
                    continue

            sorted_candidates = sorted(
                candidates, key=lambda x: x[1])  # sort by cost
            logging.debug('Candidate edges: {}'.format(sorted_candidates))

            n = len(sorted_candidates)
            if n > 0:
                best_candidates = [sorted_candidates[0]]
                j = 0
                while (j + 1 < n and
                        sorted_candidates[j][1] == sorted_candidates[j + 1][1]):
                    best_candidates.append(sorted_candidates[j + 1])
                    j += 1
            else:
                logging.info('No more edges to explore!')
                break

            for edge, dist in best_candidates:
                dists.update({edge.target.name: dist})
                # TODO: improve tie breaking
                previous_nodes.update({edge.target.name: edge.source.name})
                logging.info('Edge selected: {}-->{} ({})'.format(
                    edge.source.name, edge.target.name, dist))

            logging.debug('len(dists): {}, len(graph.vertices): {}'.format(
                len(dists), len(graph.vertices)))

            i += 1

        if target_name:  # a target node was provided
            shortest_path = _extract_path(previous_nodes, target_name)
            shortest_distance = dists[target_name]
            return shortest_distance, shortest_path

        else:
            return dists, previous_nodes

    except ValueError as e:
        logging.error(e)


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

    result = {'status': None,
              'optimal_value': None,
              'optimal_solution': None, }

    global z_star

    if (lp_status != 'optimal' or
            lp_value <= z_star):  # fathom: lp infeasible or suboptimal
        result.update(
            {'status': lp_status,
             'optimal_value': -np.inf, })

    elif is_integer_solution(
            lp_solution, epsilon=tolerance):  # fathom: integer solution found
        z_star = max(z_star, lp_value)
        result.update(
            {'status': lp_status,
             'optimal_value': lp_value,
             'optimal_solution': [int(x) for x in lp_solution], })

    elif var_index >= len(bip.variables()):  # fathom: reached a leaf node
        result.update(
            {'status': 'optimal',
             'optimal_value': bip.objective.value,
             'optimal_solution': [int(v.value[0]) for v in bip.variables()], })
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
