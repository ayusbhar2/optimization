import cvxpy as cp

import logging
import unittest

from solver.algorithms import branch_and_bound, get_shortest_path
from solver.algorithms import _extract_path, get_minimum_spanning_tree
from solver.classes import BinaryIntegerProblem, Edge, Graph, Vertex
from solver.utils import is_integer_solution, get_variable


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


class TestVertex(unittest.TestCase):

    def test_init(self):
        v = Vertex('a', attr1=1, attr2=2)
        self.assertEqual(v.attr1, 1)
        self.assertEqual(v.attr2, 2)

        v.add_attr(delta='dawn')
        self.assertEqual(v.delta, 'dawn')


class TestEdge(unittest.TestCase):

    def test_init(self):
        e1 = Edge(1, 2, **{'cost': 1})
        e2 = Edge(1, 2, **{'cost': 2})
        self.assertEqual(e1, e2)
        self.assertEqual(e1.cost, 1)
        self.assertTrue(isinstance(e1.source, Vertex))

        e3 = Edge(Vertex(1), Vertex(2), cost=1)
        e4 = Edge(Vertex(1), Vertex(2), cost=2)
        self.assertEqual(e3, e4)
        self.assertEqual(e3.cost, 1)
        self.assertTrue(isinstance(e3.source, Vertex))

        e5 = Edge('1', '2', cost=1)
        e6 = Edge('1', '2', cost=2)
        self.assertEqual(e5, e6)
        self.assertEqual(e5.cost, 1)
        self.assertTrue(isinstance(e5.source, Vertex))

    def test_update(self):
        e1 = Edge(1, 2, cost=1)
        e1.update(dist=4)
        self.assertEqual(e1.dist, 4)

    def test_delete(self):
        e1 = Edge(1, 2, cost=1)
        e1.delete_attr('cost')
        self.assertFalse('cost' in e1.__dict__)


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

        self.assertEqual(
            set([v.name for v in graph.vertices]), {0, 1, 2, 3})

    def test_add_edge(self):
        e1 = Edge(0, 1, cost=4)
        edge_list = [e1]
        graph = Graph(edge_list)

        with self.assertRaises(ValueError):
            graph.add_edge(e1)

        self.assertIn(Vertex(0), graph.vertices)

    def test_update_edge(self):
        e1 = Edge(0, 1, cost=4)
        edge_list = [e1]
        graph = Graph(edge_list)

        graph.update_edge(0, 1, alpha='boom')
        self.assertEqual(graph.get_edge(0, 1).alpha, 'boom')

    def test_get_edges(self):
        e1 = Edge(0, 1, cost=4)
        e2 = Edge(0, 2, cost=5)
        edge_list = [e1, e2]
        graph = Graph(edge_list)

        result = graph.get_edges(source_name=0)
        self.assertIn(e1, result)
        self.assertIn(e2, result)

        result = graph.get_edges(target_name=1)
        self.assertEqual(result[0], e1)


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
        result = get_shortest_path(
            graph, 'O', target_name='T', algorithm='dijkstra')
        self.assertEqual(result[0], 13)
        self.assertIn(result[1], [['O', 'A', 'B', 'D', 'T'],
                      ['O', 'A', 'B', 'E', 'D', 'T']])  # two optimal solutions

        result = get_shortest_path(graph, 'O', target_name='C',
                                   algorithm='dijkstra')
        self.assertEqual(result[0], 4)
        self.assertEqual(result[1], ['O', 'C'])

        # without target
        result = get_shortest_path(graph, 'O', target_name=None,
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


class TestUtils(unittest.TestCase):

    def test_is_integer_solution(self):
        self.assertTrue(is_integer_solution([1e-7], 1e-7))
        self.assertFalse(is_integer_solution([1.5], 1e-7))

    def test_get_variable(self):
        x1 = cp.Variable(1, name='x1')
        x2 = cp.Variable(1, name='x2')

        obj = cp.Minimize(x1 + x2)
        constr = [x1 + x2 <= 1, x1 >= 0, x2 >= 0]
        prob = cp.Problem(obj, constr)

        v1 = get_variable(prob, 'x1')

        self.assertEqual(v1.name(), 'x1')
        self.assertEqual(v1.id, x1.id)


class TestMinimumSpanningTree(unittest.TestCase):
    def test_minimum_spanning_tree(self):
        edge_list = [
            Edge('O', 'A', length=2),
            Edge('O', 'B', length=5),
            Edge('O', 'C', length=4),
            Edge('A', 'O', length=2),
            Edge('A', 'B', length=2),
            Edge('A', 'D', length=7),
            Edge('B', 'O', length=5),
            Edge('B', 'A', length=2),
            Edge('B', 'C', length=1),
            Edge('B', 'D', length=4),
            Edge('B', 'E', length=3),
            Edge('C', 'O', length=4),
            Edge('C', 'B', length=1),
            Edge('C', 'E', length=4),
            Edge('D', 'A', length=7),
            Edge('D', 'B', length=4),
            Edge('D', 'E', length=1),
            Edge('D', 'T', length=5),
            Edge('E', 'B', length=3),
            Edge('E', 'C', length=4),
            Edge('E', 'D', length=1),
            Edge('E', 'T', length=7),
            Edge('T', 'D', length=5),
            Edge('T', 'E', length=7),
        ]

        graph = Graph(edge_list)
        tree = get_minimum_spanning_tree(graph)
        self.assertEqual(len(tree.edges), 6)

        self.assertTrue(
            graph.get_edge('O', 'A') in tree.edges or
            graph.get_edge('A', 'O') in tree.edges
        )
        self.assertTrue(
            graph.get_edge('B', 'A') in tree.edges or
            graph.get_edge('A', 'B') in tree.edges
        )
        self.assertTrue(
            graph.get_edge('B', 'C') in tree.edges or
            graph.get_edge('C', 'B') in tree.edges
        )
        self.assertTrue(
            graph.get_edge('B', 'E') in tree.edges or
            graph.get_edge('E', 'B') in tree.edges
        )
        self.assertTrue(
            graph.get_edge('D', 'E') in tree.edges or
            graph.get_edge('E', 'D') in tree.edges
        )
        self.assertTrue(
            graph.get_edge('D', 'T') in tree.edges or
            graph.get_edge('T', 'D') in tree.edges
        )
        self.assertIsNone(
            get_minimum_spanning_tree(Graph()))


if __name__ == '__main__':

    logging.basicConfig(level=logging.ERROR)
    unittest.main()
