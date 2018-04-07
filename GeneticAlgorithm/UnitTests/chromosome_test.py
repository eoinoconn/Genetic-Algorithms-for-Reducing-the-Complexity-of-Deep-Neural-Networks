import unittest
from GeneticAlgorithm.chromosome import *


class TestChromosome(unittest.TestCase):

#    def setUp(self):

    def setup_chromosome_test(self):
        node = ConvNode()
        chromo = Chromosome()
        chromo.add_node(node)
        chromo.add_vertex(node, node.id)
        print(chromo)

    def build_dense_model_test(self):
        chromo = Chromosome()
        chromo.build().summary()
        print("test 2")

    def build_conv_model_test(self):
        chromo = Chromosome()
        chromo.add_random_conv_node()
        chromo.build().summary()

if __name__ == '__main__':

    unittest.main()
