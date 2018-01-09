import unittest
from Python.GeneticAlgorithm.genes import Genes, LAYER_DEPTH, MAX_LAYERS


class TestGenes(unittest.TestCase):

    def setUp(self):
        self.chromo = Genes()

    def test_len(self):
        self.setup_fake_chromosome()
        self.assertEquals(7, self.chromo.__len__())

    def testAddLayer(self):
        # Add the layer values
        layer = [i for i in range(0, LAYER_DEPTH)]
        self.chromo.add_layer(layer)

        # Create expected Layer
        expected_chromo = [[0 for x in range(LAYER_DEPTH)] for y in range(MAX_LAYERS)]
        expected_chromo[0] = layer

        self.assertEqual(expected_chromo.__str__(), self.chromo.__str__())

    def testRemoveLayer(self):
        # Create expected Layer
        expected_chromo = [[0 for x in range(LAYER_DEPTH)] for y in range(MAX_LAYERS)]

        # Add the layer values
        for x in range(1, 5):
            layer = [i*x for i in range(1, LAYER_DEPTH+1)]
            self.chromo.add_layer(layer)
            # add layers to expected chromosome except last layer
            if x < 5:
                expected_chromo[x-1] = layer

        self.assertEqual(expected_chromo.__str__(), self.chromo.__str__())

    def testBuildModelDense(self):
        layer = [0 for i in range(0, LAYER_DEPTH)]
        layer[0] = 1    # Dense layer
        layer[1] = 64   # 64 units
        layer[2] = 1    # input layer
        self.chromo.add_layer(layer)

        layer = [0 for i in range(0, LAYER_DEPTH)]
        layer[0] = 1
        layer[1] = 32
        self.chromo.add_layer(layer)

        layer = [0 for i in range(0, LAYER_DEPTH)]
        layer[0] = 1
        layer[1] = 10
        self.chromo.add_layer(layer)
        self.chromo.build_model()

    def testBuildModel2DConv(self):
        layer = [0 for i in range(0, LAYER_DEPTH)]
        layer[0] = 2  # Conv layer
        layer[1] = 64  # 64 units
        layer[2] = 1  # input layer
        layer[3] = 2    # slide size
        layer[4] = 'relu'
        self.chromo.add_layer(layer)

        layer = [0 for i in range(0, LAYER_DEPTH)]
        layer[0] = 2
        layer[1] = 32
        layer[3] = 2
        layer[4] = 'relu'
        self.chromo.add_layer(layer)

        layer = [0 for i in range(0, LAYER_DEPTH)]
        layer[0] = 2
        layer[1] = 16
        layer[3] = 2
        layer[4] = 'relu'
        self.chromo.add_layer(layer)
        self.chromo.build_model()

    def setup_fake_chromosome(self):
        # add 3 dense layers
        for i in range(0, 3):
            layer = [0 for x in range(0, LAYER_DEPTH)]
            layer[0] = 2    # convolutional layer
            layer[1] = 64   # 64 units
            layer[2] = 1    # input layer
            self.chromo.add_layer(layer, i)

        # add a flatten layer
        layer = [0 for x in range(0, LAYER_DEPTH)]
        layer[0] = 3
        self.chromo.add_layer(layer, 3)

        # add 3 convolutional layers
        for i in range(0, 3):
            layer = [0 for x in range(0, LAYER_DEPTH)]
            layer[0] = 1  # dense layer
            layer[1] = 64  # 64 units
            layer[2] = 1  # input layer
            self.chromo.add_layer(layer, i+4)


if __name__ == '__main__':

    unittest.main()
