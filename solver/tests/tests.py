import cvxopt as cv
import cvxpy as cp
import numpy as np

import logging
import unittest

from solver.algorithms import branch_and_bound, get_shortest_paths, _extract_path
from solver.classes import BinaryIntegerProblem, Graph
from solver.utils import is_integer_solution


class TestBinaryIntegerProblem(unittest.TestCase):

    def test_lp_relaxation(self):
        # Original BIP (book section 11.6)
        x1 = cp.Variable(1, boolean=True, name='x1')
        x2 = cp.Variable(1, boolean=True, name='x2')
        x3 = cp.Variable(1, boolean=True, name='x3')
        x4 = cp.Variable(1, boolean=True, name='x4')

        obj = cp.Maximize(9 * x1 + 5 * x2 + 6 * x3 + 4 * x4)

        constraints = [6 * x1 + 3 * x2 + 5 * x3 + 2 * x4 <= 10,
                                             x3 +     x4 <= 1,
                          -x1              + x3          <= 0,
                                   -x2              + x4 <= 0]

        bip = BinaryIntegerProblem(obj, constraints)
        result = bip.solve_lp_relaxation()

        self.assertAlmostEqual(x1.value[0], 0.8333333)
        self.assertAlmostEqual(x2.value[0], 1)
        self.assertAlmostEqual(x3.value[0], 0)
        self.assertAlmostEqual(x4.value[0], 1)

        self.assertAlmostEqual(result['optimal_value'], 16.5)

        self.assertTrue(x1.attributes['boolean'])
        self.assertTrue(x2.attributes['boolean'])
        self.assertTrue(x3.attributes['boolean'])
        self.assertTrue(x4.attributes['boolean'])

    def test_boolean_false_raises_exception(self):
        # Original BIP (book section 11.6)
        x1 = cp.Variable(1, boolean=False, name='x1')
        obj = cp.Maximize(x1 ** 2)
        constraints = [x1 <= 1]

        with self.assertRaises(ValueError):
            bip = BinaryIntegerProblem(obj, constraints)

class TestBranchAndBound(unittest.TestCase):

    def test_branch_and_bound(self):
        # Original BIP (book section 11.6)
        x1 = cp.Variable(1, boolean=True, name='x1')
        x2 = cp.Variable(1, boolean=True, name='x2')
        x3 = cp.Variable(1, boolean=True, name='x3')
        x4 = cp.Variable(1, boolean=True, name='x4')

        obj = cp.Maximize(9 * x1 + 5 * x2 + 6 * x3 + 4 * x4)

        constraints = [6 * x1 + 3 * x2 + 5 * x3 + 2 * x4 <= 10,
                                             x3 +     x4 <= 1,
                          -x1              + x3          <= 0,
                                   -x2              + x4 <= 0]

        bip = BinaryIntegerProblem(obj, constraints)
        result = branch_and_bound(bip)

        self.assertEqual(result.get('status'), 'optimal')
        self.assertAlmostEqual(result.get('optimal_value'), 14)
        self.assertEqual(result.get('optimal_solution')[0], 1)
        self.assertEqual(result.get('optimal_solution')[1], 1)
        self.assertEqual(result.get('optimal_solution')[2], 0)
        self.assertEqual(result.get('optimal_solution')[3], 0)

class TestUtils(unittest.TestCase):

    def test_is_integer_solution(self):
        self.assertTrue(is_integer_solution([1e-7], 1e-7))
        self.assertFalse(is_integer_solution([1.5], 1e-7))

class TestGraph(unittest.TestCase):

    def test_init_(self):
        g = {0: {1: 1, 2: 2, 3: 3},
             1: {2: 4, 3: 5},
             2: {0: 6},
             3: {2: 7}}

        graph = Graph(g)
        self.assertEqual(graph[0][1], 1)
        self.assertEqual(graph[0][2], 2)
        self.assertEqual(graph[0][3], 3)

        self.assertEqual(graph.edge_costs[(0, 1)], 1)
        self.assertEqual(graph.edge_costs[(3, 2)], 7)

class TestDijkstra(unittest.TestCase):

    def test_extract_path(self):
        previous_nodes = {
            'O': None,
            'A': 'O',
            'B': 'A',
            'C': 'O',
            'E': 'B',
            'D': 'B',
            'T': 'D',
        }
        result = _extract_path(previous_nodes, 'O')
        self.assertEqual(result, 'O')
        result = _extract_path(previous_nodes, 'A')
        self.assertEqual(result, 'OA')
        result = _extract_path(previous_nodes, 'B')
        self.assertEqual(result, 'OAB')
        result = _extract_path(previous_nodes, 'C')
        self.assertEqual(result, 'OC')
        result = _extract_path(previous_nodes, 'D')
        self.assertEqual(result, 'OABD')
        result = _extract_path(previous_nodes, 'E')
        self.assertEqual(result, 'OABE')
        result = _extract_path(previous_nodes, 'T')
        self.assertEqual(result, 'OABDT')

    def test_get_shortest_paths_with_target(self):
        # Seervada Park network
        g = {'O': {'A': 2, 'B': 5, 'C': 4},
             'A': {'O': 2, 'B': 2, 'D': 7},
             'B': {'O': 5, 'A': 2, 'C': 1, 'D': 4, 'E': 3},
             'C': {'O': 4, 'B': 1, 'E': 4},
             'D': {'A': 7, 'B': 4, 'E': 1, 'T': 5},
             'E': {'B': 3, 'C': 4, 'D': 1, 'T': 7},
             'T':{'D': 5, 'E': 7},}

        graph = Graph(g)

        # with target
        result = get_shortest_paths(graph, 'O', target='T', algorithm='dijkstra')
        self.assertEqual(result[0], 13)
        self.assertEqual(result[1], 'OABEDT')

        result = get_shortest_paths(graph, 'O', target='C', algorithm='dijkstra')
        self.assertEqual(result[0], 4)
        self.assertEqual(result[1], 'OC')

        # without target
        result = get_shortest_paths(graph, 'O', target=None, algorithm='dijkstra')
        self.assertEqual(result[0]['O'], 0)
        self.assertEqual(result[0]['A'], 2)
        self.assertEqual(result[0]['B'], 4)
        self.assertEqual(result[0]['C'], 4)
        self.assertEqual(result[0]['D'], 8)
        self.assertEqual(result[0]['E'], 7)
        self.assertEqual(result[0]['T'], 13)

        self.assertEqual(result[1]['O'], None)
        self.assertEqual(result[1]['A'], 'O')
        self.assertEqual(result[1]['B'], 'A')
        self.assertEqual(result[1]['C'], 'O')
        self.assertEqual(result[1]['D'], 'E')
        self.assertEqual(result[1]['E'], 'B')
        self.assertEqual(result[1]['T'], 'D')
        



if __name__ == '__main__':

    unittest.main()
