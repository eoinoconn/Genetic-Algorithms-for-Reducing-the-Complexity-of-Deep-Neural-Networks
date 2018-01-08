import unittest
from Python.GeneticAlgorithm.mutate import *
from Python.GeneticAlgorithm.genes import *


class TestMutate(unittest.TestCase):

    def setUp(self):
        self.chromosome = Genes()

    def test_create_parent(self):
        chromosome = create_parent()
        self.assertEquals(chromosome.num_conv(), 1)
        self.assertEquals(chromosome.num_dense(), 1)

    def test_remove_layer(self):
        self.setup_fake_chromosome()
        remove_layer(self.chromosome)
        self.assertEquals(6, self.chromosome.__len__())

    def setup_fake_chromosome(self):
        # add 3 dense layers
        for i in range(0, 3):
            layer = [0 for x in range(0, LAYER_DEPTH)]
            layer[0] = 2    # convolutional layer
            layer[1] = 64   # 64 units
            layer[2] = 1    # input layer
            self.chromosome.add_layer(layer, i)

        # add a flatten layer
        layer = [0 for x in range(0, LAYER_DEPTH)]
        layer[0] = 3
        self.chromosome.add_layer(layer, 3)

        # add 3 convolutional layers
        for i in range(0, 3):
            layer = [0 for x in range(0, LAYER_DEPTH)]
            layer[0] = 1  # dense layer
            layer[1] = 64  # 64 units
            layer[2] = 1  # input layer
            self.chromosome.add_layer(layer, i+4)

