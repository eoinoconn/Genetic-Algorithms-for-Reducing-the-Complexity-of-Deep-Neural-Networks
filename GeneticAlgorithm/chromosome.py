from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, AveragePooling2D, \
    concatenate, Input, BatchNormalization, Activation
from keras.models import Model
import configparser
import logging
import random
import csv


class Node(object):

    _id = 1

    def __init__(self):
        self.__vertex_type = ""
        self.__encoding = []
        self.__id = Node._id
        self.__active = True
        self.__logger = logging.getLogger('geneset')
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
    Node to simulate convolutional layer

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
       9   batch normalisation (Default 0 = None)
    :return:
    """

    def __init__(self, random_node=False):
        super().__init__()
        self._vertex_type = "conv"

        self.random_conv_filter_num()
        self.random_conv_kernel()
        self.random_conv_stride()
        self.encoding[3] = 'relu'
        self.random_conv_layer_padding()

        if random_node:
            self.random_conv_dropout()
            self.random_pooling_type()
            self.random_pooling_size()
            self.random_pool_stride()
            self.toggle_batch_normalisation()

    def build(self, model):
        model = Conv2D(self.encoding[0], self.encoding[2], strides=self.encoding[1], padding=self.encoding[4])(model)
        self.__logger.info("output dimensions (%d, %d)", model.shape[1], model.shape[2])
        if self.encoding[9] == 1:  # Batch normalisation layer
            model = BatchNormalization()(model)
        if self.encoding[3] is not None:
            model = Activation(self.encoding[3])(model)
        if self.encoding[6] > 0:  # Pooling layer
            if self.encoding[6] == 1:  # max pooling
                MaxPooling2D((self.encoding[7], self.encoding[7]), strides=self.encoding[8])(model)
            else:
                AveragePooling2D((self.encoding[7], self.encoding[7]), strides=self.encoding[8])(model)
            self.__logger.info("output dimensions (%d, %d)", model.shape[1], model.shape[2])
        if self.encoding[5] > 0:  # Dropout layer
            model = Dropout(self.encoding[8])(model)
        return model

    def random_conv_filter_num(self):
        min_value, max_value, interval = self.config_min_max_interval('convolutional.layer.filter')
        self.encoding[0] = 2 ** random.randrange(min_value, max_value + 1, interval)  # sets layer units
        self.__logger.info("set con filters to %d on node %d", self.encoding[0], self.id)

    def random_conv_kernel(self):
        min_value, max_value, interval = self.config_min_max_interval('convolutional.layer.kernel')
        self.encoding[1] = random.randrange(min_value, max_value + 1, interval)
        self.__logger.info("set kernel size to %d on node %d", self.encoding[1], self.id)

    def random_conv_stride(self):
        min_value, max_value, interval = self.config_min_max_interval('convolutional.layer.stride')
        self.encoding[2] = random.randrange(min_value, self.encoding[2] + 1, interval)
        self.__logger.info("set conv stride to %d on node %d", self.encoding[2], self.id)

    def random_conv_layer_padding(self):
        padding_index = random.randrange(0, 2)
        if padding_index == 0:
            self.encoding[4] = 'same'
        else:
            self.encoding[4] = 'valid'
        self.__logger.info("set padding to %s on node %d", self.encoding[4], self.id)

    def random_conv_dropout(self):
        min_value, max_value, interval = self.config_min_max_interval('conv.layer.dropout')
        self.encoding[5] = (random.randrange(min_value, max_value + 1, interval)) / 10  # Set dropout probability
        self.__logger.info("set droupout to %f on node %d", self.encoding[7], self.id)

    def random_pooling_type(self):
        min_value, max_value, interval = self.config_min_max_interval('pooling.type')
        self.encoding[6] = random.randrange(min_value, max_value + 1, interval)
        self.__logger.info("set pooling type to %s on node %d", self.encoding[6], self.id)

    def random_pooling_size(self):
        min_value, max_value, interval = self.config_min_max_interval('pooling.filter')
        self.encoding[7] = random.randrange(min_value, max_value + 1, interval)
        self.__logger.info("set pooling size to %d on node %d", self.encoding[7], self.id)

    def random_pool_stride(self):
        min_value, max_value, interval = self.config_min_max_interval('convolutional.layer.pool.stride')
        self.encoding[8] = random.randrange(min_value, self.encoding[6] + 1, interval)
        self.__logger.info("set pool stride to %d on node %d", self.encoding[8], self.id)

    def toggle_batch_normalisation(self):
        if self.encoding[9] == 1:
            self.encoding[9] = 0
        else:
            self.encoding[9] = 1
        self.__logger.info("set batch normalisation to %d on node %d", self.encoding[10], self.id)


class DenseNode(Node):
    """
    Node to simulate dense layer

    # The first value of the layer array is 1 for a dense layer
     Other variables:
       0   layer units
       1   dropout
       2   activation
    :return:
    """

    def __init__(self, random_node=False):
        super().__init__()
        self._vertex_type = "dense"

        self.random_dense_units()
        self.encoding[2] = 'relu'
        if random_node:
            self.random_dense_dropout()

    def build(self, model, output_layer=False, classes=None):
        if output_layer:  # output layer
            return Dense(classes, activation='softmax')(model)
        else:  # hidden layer
            model = Dense(self.encoding[0])(model)
            if self.encoding[4] is not None:
                model = Activation(self.encoding[2])(model)
            if self.encoding[3] > 0:  # Dropout layer
                model = Dropout(self.encoding[1])(model)
            self.__logger.info("output dimensions %d", model.shape[1])
            return model

    def random_dense_units(self):
        min_value, max_value, interval = self.config_min_max_interval('dense.layer.units')
        self.encoding[0] = (random.randrange(min_value, max_value + 1, interval))     # Set dense units
        self.__logger.info("set dense units to %d on node %d", self.encoding[0], self.id)

    def random_dense_dropout(self):
        min_value, max_value, interval = self.config_min_max_interval('dense.layer.dropout')
        self.encoding[1] = (random.randrange(min_value, max_value + 1, interval)) / 10  # Set dropout probability
        self.__logger.info("set droupout to %f on node %d", self.encoding[1], self.id)


class InputNode(Node):

    def __init__(self):
        super().__init__()
        self._vertex_type = "input"

    @staticmethod
    def build(model):
        return Flatten()(model)


class FlattenNode(Node):
    def __init__(self):
        super().__init__()
        self._vertex_type = "flatten"


class OutputNode(Node):

    def __init__(self):
        super().__init__()
        self._vertex_type = "output"


class Chromosome(object):

    _id = 1

    def __init__(self):
        self.id = Chromosome._id
        self.hyperparameters = [0 for x in range(0, 25)]
        self.fitness = None
        self.accuracy = None
        self.parameters = None
        self.age = 0
        self.vertices = {}
        self.nodes = []
        Chromosome._id += 1

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

    def logging(self, generation_num):
        with open('GeneticAlgorithm/logs/trend.csv', 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow([generation_num, ',',
                                 self.id, ',',
                                 self.age, ',',
                                 self.accuracy, ',',
                                 self.fitness, ',',
                                 self.parameters, ',',
                                 len(self), ',',
                                 # chromosome.num_conv_layers(), ',',
                                 # chromosome.num_dense_layers(), ',',
                                 # chromosome.num_incep_layers(), ',',
                                 ])

    def __len__(self):
        return len(self.nodes)

    def __str__(self):
        string = ""
        for obj in self.nodes:
            string += str(obj)
        string += str(self.vertices)
        return string
