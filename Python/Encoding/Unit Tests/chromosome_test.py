import unittest
from Python.Encoding.chromosome import Chromosome, LAYER_DEPTH, MAX_LAYERS


class TestChromosome(unittest.TestCase):

    def setUp(self):
        self.chromo = Chromosome()

    def testAddLayer(self):
        # Add the layer values
        layer = [i for i in range(0, LAYER_DEPTH)]
        self.chromo.add_layer(layer)

        # Create expected Layer
        expected_chromo = [[0 for x in range(LAYER_DEPTH)] for y in range(MAX_LAYERS)]
        expected_chromo[0] = layer

        self.assertEqual(expected_chromo.__str__(), self.chromo.__str__())

    def testBuildModel(self):
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


if __name__ == '__main__':

    unittest.main()
