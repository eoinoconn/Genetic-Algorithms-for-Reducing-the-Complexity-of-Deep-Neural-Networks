import unittest
from GeneticAlgorithm.chromosome import *


class TestChromosome(unittest.TestCase):

#    def setUp(self):

    def chromosome_test(self):
        node = ConvNode()
        chromo = Chromosome()
        chromo.add_node(node)
        chromo.add_vertex(node, node.id)
        print(chromo)


if __name__ == '__main__':

    unittest.main()
