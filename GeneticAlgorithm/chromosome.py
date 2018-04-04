import configparser
import logging
import random


class Node(object):

    _id = 1

    def __init__(self):
        self.__vertex_type = ""
        self.__encoding = []
        self.__id = Node._id
        self.__active = True
        Node._id += 1

    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self, new_id):
        self.__id = new_id

    @property
    def active(self):
        return self.__active

    @active.setter
    def active(self, value):
        self.__active = value

    @property
    def vertex_type(self):
        return self.__vertex_type

    @vertex_type.setter
    def vertex_type(self, vertex_type):
        self.__vertex_type = vertex_type

    @property
    def encoding(self):
        return self.__encoding

    @encoding.setter
    def encoding(self, encoding):
        if encoding != list(encoding):
            raise TypeError("Encoding must be of type list")
        else:
            self.__encoding = encoding

    @encoding.deleter
    def encoding(self):
        del self.__encoding

    @staticmethod
    def config_min_max_interval(config_name):
        config = configparser.ConfigParser()
        config.read('GeneticAlgorithm/Config/training_parameters.ini')
        config = config[config_name]
        minimum = int(config['minimum'])
        maximum = int(config['maximum'])
        interval = int(config['interval'])
        return minimum, maximum, interval

    def __str__(self):
        enc_string = "Node type {}, Node id {}\n".format(self.vertex_type, self._id)
        enc_string += str(self.encoding)
        return enc_string

    def __hash__(self):
        return hash(self._id)


class ConvNode(Node):
    """
    returns a randomly generated convolutional layer

    The first value of the layer array is 2 for a convolutional layer
    Other variables:
       0   layer units
       1   stride
       2   kernal size
       3   activation
       4   Conv layer padding
       5   Dropout
       6   pooling type(Default 0 = None)
       7   pool size
       8   Pool stride
       9  batch normalisation (Default 0 = None)
    :return:
    """

    def __init__(self, random=False):
        super().__init__()
        self._vertex_type = "conv"

        if random:
            self.random_conv_filter_num()
            self.random_conv_kernel()
            self.random_conv_stride()
            self.encoding[3] = 'relu'
            self.random_conv_layer_padding()
            self.random_conv_dropout()
            self.random_pooling_type()
            self.random_pooling_size()
            self.random_pool_stride()
            self.toggle_batch_normalisation()

    def random_conv_filter_num(self):
        min_value, max_value, interval = self.config_min_max_interval('convolutional.layer.filter')
        self.encoding[0] = 2 ** random.randrange(min_value, max_value + 1, interval)  # sets layer units

    def random_conv_kernel(self):
        min_value, max_value, interval = self.config_min_max_interval('convolutional.layer.kernel')
        self.encoding[1] = random.randrange(min_value, max_value + 1, interval)

    def random_conv_stride(self):
        min_value, max_value, interval = self.config_min_max_interval('convolutional.layer.stride')
        self.encoding[2] = random.randrange(min_value, self.encoding[2] + 1, interval)

    def random_conv_layer_padding(self):
        padding_index = random.randrange(0, 2)
        if padding_index == 0:
            self.encoding[4] = 'same'
        else:
            self.encoding[4] = 'valid'

    def random_conv_dropout(self):
        min_value, max_value, interval = self.config_min_max_interval('conv.layer.dropout')
        self.encoding[5] = (random.randrange(min_value, max_value + 1, interval)) / 10  # Set dropout probability
        logger.info("set droupout to %f on node %d", self.encoding[7], self.id)


    def random_pooling_type(layer):
        min_value, max_value, interval = config_min_max_interval('pooling.type')
        layer[5] = random.randrange(min_value, max_value + 1, interval)
        return layer

    def random_pooling_size(layer):
        min_value, max_value, interval = config_min_max_interval('pooling.filter')
        layer[6] = random.randrange(min_value, max_value + 1, interval)
        return layer

    def random_pool_stride(self):
        min_value, max_value, interval = config_min_max_interval('convolutional.layer.pool.stride')
        layer[8] = random.randrange(min_value, layer[6] + 1, interval)
        return layer

    def toggle_batch_normalisation(self):
        config = get_config()
        if not config['batch.normalisation'].getboolean('enabled'):
            logger.info("batch normalisation disabled")
            return False
        while True:
            layer_index = random.randrange(0, genes.__len__())
            layer = genes.get_layer(layer_index)
            if layer[0] == 2:   # check if conv layer
                old_layer = layer
                if layer[10] == 1:
                    layer[10] = 0
                else:
                    layer[10] = 1
                genes.overwrite_layer(layer, layer_index)
                if check_valid_geneset(genes, logger):
                    logger.info("toggling batch normalisation to layer %d", layer_index)
                    return True
                else:
                    logger.info("toggling batch normalisation to layer %d failed", layer_index)
                    genes.overwrite_layer(old_layer, layer_index)
                    return False


class DenseNode(Node):

    def __init__(self):
        super().__init__()
        self._vertex_type = "dense"


class Input(Node):

    def __init__(self):
        super().__init__()
        self._vertex_type = "input"


class Flatten(Node):
    def __init__(self):
        super().__init__()
        self._vertex_type = "flatten"


class Output(Node):

    def __init__(self):
        super().__init__()
        self._vertex_type = "output"


class Chromosome(object):

    def __init__(self):
        self.hyperparameters = []
        self.vertices = {}
        self.nodes = []

        config = configparser.ConfigParser()
        config.read('GeneticAlgorithm/Config/training_parameters.ini')
        logger = logging.getLogger('Chromosome')
        logger.info("creating parent genes")

    def add_node(self, node):
        self.nodes.append(node)
        if isinstance(node, Node):
            self.vertices[node] = []
        else:
            raise TypeError("node should be of type Node")

    def add_vertex(self, node, vertex):
        self.vertices[node].append(vertex)

    def remove_node(self, node):
        self.vertices[node][0].actve = False

    def remove_vertex(self, node, vertex):
        if vertex in self.vertices[node]:
            self.vertices[node].remove(vertex)

    def __str__(self):
        string = ""
        for obj in self.nodes:
            string += str(obj)
        string += str(self.vertices)
        return string

