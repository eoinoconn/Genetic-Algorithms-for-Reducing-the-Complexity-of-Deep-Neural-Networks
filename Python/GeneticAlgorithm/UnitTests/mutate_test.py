import unittest
from Python.GeneticAlgorithm.mutate import *
from Python.GeneticAlgorithm.genes import *


class TestMutate(unittest.TestCase):

    def setUp(self):
        self.chromosome = Genes()

    def test_remove_layer(self):
        self.setup_chromosome()
        remove_layer(self.chromosome)
        self.assertEquals(6, self.chromosome.__len__())

    def test_check_valid_padding_bad(self):
        self.setup_bad_chromosome()
        self.assertFalse(check_valid_pooling(self.chromosome))

    def test_check_valid_padding_good(self):
        self.setup_good_chromosome()
        self.assertTrue(check_valid_pooling(self.chromosome))

    def setup_good_chromosome(self):
        # add 3 dense layers
        for i in range(0, 3):
            layer = [0 for x in range(0, LAYER_DEPTH)]
            layer[0] = 2  # convolutional layer
            layer[1] = 64  # 64 units
            layer[2] = 1  # input layer
            layer[3] = 2+i # window size
            layer[5] = 1
            layer[6] = 2
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
            self.chromosome.add_layer(layer, i + 4)

    def setup_bad_chromosome(self):
        # add 3 dense layers
        for i in range(0, 3):
            layer = [0 for x in range(0, LAYER_DEPTH)]
            layer[0] = 2  # convolutional layer
            layer[1] = 64  # 64 units
            layer[2] = 1  # input layer
            layer[3] = 3+i # window size
            layer[5] = 1
            layer[6] = 2
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
            self.chromosome.add_layer(layer, i + 4)

    def setup_chromosome(self):
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

