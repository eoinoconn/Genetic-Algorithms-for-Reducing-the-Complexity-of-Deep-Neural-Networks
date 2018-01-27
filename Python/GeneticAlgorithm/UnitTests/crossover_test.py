import unittest
from Python.GeneticAlgorithm.crossover import crossover
from Python.GeneticAlgorithm.genes import Genes, LAYER_DEPTH


def setup_fake_chromosome(random):
    chromo = Genes()

    # add 3 dense layers
    for i in range(0, 3):
        layer = [0 for x in range(0, LAYER_DEPTH)]
        layer[0] = 2    # convolutional layer
        layer[1] = 8*random*i   # 64 units
        layer[2] = 1    # input layer
        chromo.add_layer(layer, i)

    # add a flatten layer
    layer = [0 for x in range(0, LAYER_DEPTH)]
    layer[0] = 3
    chromo.add_layer(layer, 3)

    # add 3 convolutional layers
    for i in range(0, 3):
        layer = [0 for x in range(0, LAYER_DEPTH)]
        layer[0] = 1  # dense layer
        layer[1] = 4*random*i  # 64 units
        layer[2] = 1  # input layer
        chromo.add_layer(layer, i+4)

    return chromo


class TestCrossover(unittest.TestCase):

    def setUp(self):
        self.parent_1 = setup_fake_chromosome(2)
        self.parent_2 = setup_fake_chromosome(3)

    def crossover_test(self):
        child = crossover(self.parent_1, self.parent_2)
        print(child)


if __name__ == '__main__':

    unittest.main()
