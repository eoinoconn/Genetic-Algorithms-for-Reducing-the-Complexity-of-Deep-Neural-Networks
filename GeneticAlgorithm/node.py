from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, AveragePooling2D, \
    concatenate, Input, BatchNormalization, Activation, Concatenate
from keras.models import Model
import configparser
from pathlib import Path
import logging
import random
import csv


class DimensionException(Exception):
    pass


class CantAddNode(Exception):
    pass


class GeneticObject(object):

    @staticmethod
    def config_min_max_interval(config_name):

        config = configparser.ConfigParser()
        config.read("../training_parameters.ini")
        config = config[config_name]
        minimum = int(config['minimum'])
        maximum = int(config['maximum'])
        interval = int(config['interval'])
        return minimum, maximum, interval


class Node(GeneticObject):

    _id = 1

    def __init__(self):
        self.__vertex_type = ""
        self.__encoding = [0 for x in range(0, 12)]
        self._id = Node._id
        self.active = True
        self.model = None
        self._logger = logging.getLogger('geneset')
        Node._id += 1

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, new_id):
        self._id = new_id

    @property
    def output_dimension(self):
        return int(self.model.get_shape()[1])

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

    def delete_model(self):
        del self.model
        self.model = None

    def is_built(self):
        if self.model is None:
            return False
        else:
            return True

    def __str__(self):
        enc_string = "Node type {}, Node id {}\n".format(str(self.__vertex_type), self._id)
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
        self.last_mutate_index = None
        self.pre_mutate_value = None

        if random_node:
            self.random_conv_dropout()
            self.random_pooling_type()
            self.random_pooling_size()
            self.random_pool_stride()
            self.toggle_batch_normalisation()

    def build(self, model):
        if self.compute_output_dimension((model.get_shape()[1], model.get_shape()[2])) < 1:
            self._logger.info("Dimensions are {}" + str(self.compute_output_dimension((model.get_shape()[1], model.get_shape()[2]))))
            raise DimensionException("Error with dimensions; model shape {}", model.get_shape())
        model = Conv2D(self.encoding[0], self.encoding[2], strides=self.encoding[1], padding=self.encoding[4])(model)
        if self.encoding[9] == 1:  # Batch normalisation layer
            model = BatchNormalization()(model)
        if self.encoding[3] is not None:
            model = Activation(self.encoding[3])(model)
        if self.encoding[6] > 0:  # Pooling layer
            if self.encoding[6] == 1:  # max pooling
                MaxPooling2D((self.encoding[7], self.encoding[7]), strides=self.encoding[8])(model)
            else:
                AveragePooling2D((self.encoding[7], self.encoding[7]), strides=self.encoding[8])(model)
        if self.encoding[5] > 0:  # Dropout layer
            model = Dropout(self.encoding[8])(model)
        self._logger.info("Dimensions after build of node %d are {}", self.id)
        self._logger.info(model.get_shape())
        self.model = model

    def compute_output_dimension(self, input_dimensions):
            """Computes expected conv node output based on encoding"""
            width, heigth = input_dimensions
            if width != heigth:
                raise ValueError("width and height are not the same")
            if self.encoding[4] == "same":
                output_of_conv = int(width)/self.encoding[1]
            else:
                output_of_conv = int(int(width - self.encoding[2]) / int(self.encoding[1]))
            self._logger.info("output dimensions of convolutional layer for node {0} is {1}".format(str(self.id), str(
                output_of_conv)))
            if self.encoding[6] > 0:
                output_dimension = int((output_of_conv - self.encoding[7]) / self.encoding[8] + 1)
            else:
                output_dimension = output_of_conv
            self._logger.info("output dimensions of pooling layer for node {0} is {1}".format(str(self.id), str(
                output_dimension)))
            return output_dimension

    def mutate(self):
        rand = random.randrange(0, 9)
        if rand == 0:
            self.random_conv_filter_num()
        elif rand == 1:
            self.random_conv_kernel()
        elif rand == 2:
            self.random_conv_stride()
        elif rand == 3:
            self.random_conv_layer_padding()
        elif rand == 4:
            self.random_conv_dropout()
        elif rand == 5:
            self.random_pooling_type()
        elif rand == 6:
            self.random_pooling_size()
        elif rand == 7:
            self.random_pool_stride()
        else:
            self.toggle_batch_normalisation()

    def reshuffle_dimensions(self):
        self.random_conv_kernel()
        self.random_conv_stride()

    def random_conv_filter_num(self):
        min_value, max_value, interval = self.config_min_max_interval('convolutional.layer.filter')
        self.encoding[0] = 2 ** random.randrange(min_value, max_value + 1, interval)  # sets layer units
        self._logger.info("set con filters to %d on node %d", self.encoding[0], self.id)

    def random_conv_kernel(self):
        self.last_mutate_index = 1
        self.pre_mutate_value = self.encoding[1]
        min_value, max_value, interval = self.config_min_max_interval('convolutional.layer.kernel')
        self.encoding[1] = random.randrange(min_value, max_value + 1, interval)
        self._logger.info("set kernel size to %d on node %d", self.encoding[1], self.id)

    def random_conv_stride(self):
        self.last_mutate_index = 2
        self.pre_mutate_value = self.encoding[2]
        min_value, max_value, interval = self.config_min_max_interval('convolutional.layer.stride')
        self.encoding[2] = random.randrange(min_value, self.encoding[1] + 1, interval)
        self._logger.info("set conv stride to %d on node %d", self.encoding[2], self.id)

    def random_conv_layer_padding(self):
        # padding_index = random.randrange(0, 2)
        # if padding_index == 0:
        #     self.encoding[4] = 'same'
        # else:
        #     self.encoding[4] = 'valid'
        self.encoding[4] = 'same'
        self._logger.info("set padding to %s on node %d", self.encoding[4], self.id)

    def random_conv_dropout(self):
        min_value, max_value, interval = self.config_min_max_interval('conv.layer.dropout')
        self.encoding[5] = (random.randrange(min_value, max_value + 1, interval)) / 10  # Set dropout probability
        self._logger.info("set droupout to %f on node %d", self.encoding[7], self.id)

    def random_pooling_type(self):
        self.last_mutate_index = 7
        self.pre_mutate_value = self.encoding[7]
        min_value, max_value, interval = self.config_min_max_interval('pooling.type')
        self.encoding[6] = random.randrange(min_value, max_value + 1, interval)
        self._logger.info("set pooling type to %s on node %d", self.encoding[6], self.id)

    def random_pooling_size(self):
        self.last_mutate_index = 7
        self.pre_mutate_value = self.encoding[7]
        min_value, max_value, interval = self.config_min_max_interval('pooling.filter')
        self.encoding[7] = random.randrange(min_value, max_value + 1, interval)
        self._logger.info("set pooling size to %d on node %d", self.encoding[7], self.id)

    def random_pool_stride(self):
        self.last_mutate_index = 8
        self.pre_mutate_value = self.encoding[8]
        min_value, max_value, interval = self.config_min_max_interval('convolutional.layer.pool.stride')
        self.encoding[8] = random.randrange(min_value, self.encoding[6] + 1, interval)
        self._logger.info("set pool stride to %d on node %d", self.encoding[8], self.id)

    def toggle_batch_normalisation(self):
        self.last_mutate_index = 9
        self.pre_mutate_value = self.encoding[9]
        if self.encoding[9] == 1:
            self.encoding[9] = 0
        else:
            self.encoding[9] = 1
        self._logger.info("set batch normalisation to %d on node %d", self.encoding[10], self.id)

    def undo_last_mutate(self):
        self.encoding[self.last_mutate_index] = self.pre_mutate_value


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
            self._logger.info("output dimensions %d", model.shape[1])
            return model

    def mutate(self):
        rand = random.randrange(0, 3)
        if rand == 0:
            self.random_dense_units()
        else:
            self.random_dense_dropout()

    def random_dense_units(self):
        min_value, max_value, interval = self.config_min_max_interval('dense.layer.units')
        self.encoding[0] = 2 ** (random.randrange(min_value, max_value + 1, interval))     # Set dense units
        self._logger.info("set dense units to %d on node %d", self.encoding[0], self.id)

    def random_dense_dropout(self):
        min_value, max_value, interval = self.config_min_max_interval('dense.layer.dropout')
        self.encoding[1] = (random.randrange(min_value, max_value + 1, interval)) / 10  # Set dropout probability
        self._logger.info("set droupout to %f on node %d", self.encoding[1], self.id)


class ConvInputNode(Node):

    def __init__(self, shape):
        super().__init__()
        self._vertex_type = "input"
        self.shape = shape

    def build(self):
        self.model = Input(self.shape)
        return self.model

    @property
    def output_dimension(self):
        return self.shape[1]


class ConvOutputNode(Node):

    def __init__(self):
        super().__init__()
        self._vertex_type = "output"

    def build(self, model):
        self.model = Flatten()(model)

    @property
    def output_dimension(self):
        return self.model.get_shape()[1]

