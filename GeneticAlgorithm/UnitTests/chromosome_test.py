import unittest
from GeneticAlgorithm.chromosome import *


class TestChromosome(unittest.TestCase):

#    def setUp(self):

    def setup_chromosome_test(self):
        node = ConvNode()
        chromo = Chromosome((32, 32, 3))
        chromo.add_node(node)
        chromo.add_vertex(node, node.id)
        print(chromo)

    @staticmethod
    def build_basic_model_test():
        chromo = Chromosome((32, 32, 3))
        chromo.build().summary()

    def dense_model_test(self):
        for i in range(50):
            self.build_dense_model()
        assert True

    def build_dense_model(self):
        chromo = Chromosome((32, 32, 3))
        for i in range(10):
            chromo.add_random_dense_node()
        chromo.build().summary()

    def simple_conv_model_test(self):
        counter = 0
        for i in range(20):
            if self.build_simple_conv_model():
                counter += 1
        print("failed conv models = " + str(counter))
        assert True

    def complex_conv_model_test(self):
        counter = 0
        for i in range(10):
            if self.build_complex_conv_model():
                counter += 1
        print("failed conv models = " + str(counter))
        assert True

    def real_model_test(self):
        counter = 0
        for i in range(10):
            if self.build_real_model():
                counter += 1
        print("failed conv models = " + str(counter))
        assert True

    @staticmethod
    def build_simple_conv_model():
        try:
            chromo = Chromosome((32, 32, 3))
            chromo.add_random_conv_node()
            chromo.build().summary()
        except DimensionException:
            return True
            pass

    @staticmethod
    def build_complex_conv_model():
        try:
            chromo = Chromosome((32, 32, 3))
            for i in range(5):
                chromo.add_random_conv_node()
            chromo.build().summary()
        except CantAddNode:
            del chromo
            return True

    @staticmethod
    def build_real_model():
        try:
            chromo = Chromosome((32, 32, 3))
            for i in range(5):
                chromo.add_random_conv_node()
            for i in range(5):
                chromo.add_random_dense_node()
            chromo.build().summary()
        except CantAddNode:
            return True

    #def mutate_test(self):
    #    chromo = Chromosome((32, 32, 3))
    #    for i in range(30):
    #        chromo.mutate()

if __name__ == '__main__':

    unittest.main()
