import cvxpy as cp

import logging
import unittest

from solver.algorithms import branch_and_bound, get_shortest_path
from solver.algorithms import _extract_path
from solver.classes import BinaryIntegerProblem, Edge, Graph
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
            BinaryIntegerProblem(obj, constraints)


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


class TestEdge(unittest.TestCase):

    def test_init(self):
        e1 = Edge(1, 2, **{'cost': 1})
        e2 = Edge(1, 2, **{'cost': 2})
        self.assertEqual(e1, e2)
        self.assertEqual(e1.cost, 1)

    def test_update(self):
        e1 = Edge(1, 2, cost=1)
        e1.update(dist=4)
        self.assertEqual(e1.dist, 4)


class TestGraph(unittest.TestCase):

    def test_init_(self):
        edge_list = [
            Edge(0, 1, **{'cost': 1}),
            Edge(0, 2, **{'cost': 2}),
            Edge(0, 3, **{'cost': 3}),
            Edge(1, 2, **{'cost': 4}),
            Edge(1, 3, **{'cost': 5}),
            Edge(2, 0, **{'cost': 6}),
            Edge(3, 2, **{'cost': 7}), ]

        graph = Graph(edge_list)

        self.assertEqual(graph.get_edge(0, 1).cost, 1)
        self.assertEqual(graph.get_edge(0, 2).cost, 2)
        self.assertEqual(graph.get_edge(0, 3).cost, 3)

        self.assertEqual(graph.get_edge(0, 1).cost, 1)
        self.assertEqual(graph.get_edge(3, 2).cost, 7)

        self.assertEqual(graph.vertices, {0, 1, 2, 3})

    def test_add_edge(self):
        e1 = Edge(0, 1, cost=4)
        edge_list = [e1]
        graph = Graph(edge_list)

        with self.assertRaises(ValueError):
            graph.add_edge(e1)

    def test_update_edge(self):
        e1 = Edge(0, 1, cost=4)
        edge_list = [e1]
        graph = Graph(edge_list)

        graph.update_edge(0, 1, alpha='boom')
        self.assertEqual(graph.get_edge(0, 1).alpha, 'boom')


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
        self.assertEqual(result, ['O'])
        result = _extract_path(previous_nodes, 'A')
        self.assertEqual(result, ['O', 'A'])
        result = _extract_path(previous_nodes, 'B')
        self.assertEqual(result, ['O', 'A', 'B'])
        result = _extract_path(previous_nodes, 'C')
        self.assertEqual(result, ['O', 'C'])
        result = _extract_path(previous_nodes, 'D')
        self.assertEqual(result, ['O', 'A', 'B', 'D'])
        result = _extract_path(previous_nodes, 'E')
        self.assertEqual(result, ['O', 'A', 'B', 'E'])
        result = _extract_path(previous_nodes, 'T')
        self.assertEqual(result, ['O', 'A', 'B', 'D', 'T'])

    def test_get_shortest_path_with_target(self):
        # Seervada Park network
        edge_list = [
            Edge('O', 'A', cost=2),
            Edge('O', 'B', cost=5),
            Edge('O', 'C', cost=4),

            Edge('A', 'O', cost=2),
            Edge('A', 'B', cost=2),
            Edge('A', 'D', cost=7),

            Edge('B', 'O', cost=5),
            Edge('B', 'A', cost=2),
            Edge('B', 'C', cost=1),
            Edge('B', 'D', cost=4),
            Edge('B', 'E', cost=3),

            Edge('C', 'O', cost=4),
            Edge('C', 'B', cost=1),
            Edge('C', 'E', cost=4),

            Edge('D', 'A', cost=7),
            Edge('D', 'B', cost=4),
            Edge('D', 'E', cost=1),
            Edge('D', 'T', cost=5),

            Edge('E', 'B', cost=3),
            Edge('E', 'C', cost=4),
            Edge('E', 'D', cost=1),
            Edge('E', 'T', cost=7),

            Edge('T', 'D', cost=5),
            Edge('T', 'E', cost=7),
        ]

        graph = Graph(edge_list)

        # with target
        result = get_shortest_path(graph, 'O', target='T',
                                   algorithm='dijkstra')
        self.assertEqual(result[0], 13)
        self.assertIn(result[1], [['O', 'A', 'B', 'D', 'T'],
                      ['O', 'A', 'B', 'E', 'D', 'T']])  # two optimal solutions

        result = get_shortest_path(graph, 'O', target='C',
                                   algorithm='dijkstra')
        self.assertEqual(result[0], 4)
        self.assertEqual(result[1], ['O', 'C'])

        # without target
        result = get_shortest_path(graph, 'O', target=None,
                                   algorithm='dijkstra')
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
        self.assertIn(result[1]['D'], ['E', 'B'])  # two optimal solutions
        self.assertEqual(result[1]['E'], 'B')
        self.assertEqual(result[1]['T'], 'D')
        

if __name__ == '__main__':

    logging.basicConfig(level=logging.ERROR)
    unittest.main()
