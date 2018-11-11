from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, AveragePooling2D, \
    concatenate, Input, BatchNormalization, Activation, Concatenate
from keras.models import Model
import configparser
from pathlib import Path
import logging
import random
import csv
from PySearch.exceptions import DimensionException

class GeneticObject(object):
    """Base encoding object"""
    
    @staticmethod
    def config_min_max_interval(config_name):
        """
        Parses configuration for hyper-parameter boundaries.
        """
        config = configparser.ConfigParser()
        config.read("PySearch/training_parameters.ini")
        config = config[config_name]
        minimum = int(config['minimum'])
        maximum = int(config['maximum'])
        interval = int(config['interval'])
        return minimum, maximum, interval


class Node(GeneticObject):
    """Base class for chromosome nodes."""
    _id = 1
    __vertex_type = "Base Node"

    def __init__(self):
        self.__encoding = [0 for x in range(0, 12)]
        self._id = Node._id
        self._active = True
        self._model = None
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
        return int(self._model.get_shape()[1])

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, value):
        self._active = value

    @property
    def vertex_type(self):
        return self.__vertex_type

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
        self.__encoding = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, new_model):
        self._model = new_model

    @model.deleter
    def model(self):
        self._model = None

    def is_built(self):
        """Returns true of model is present"""
        if self._model is None:
            return False
        else:
            return True

    def __str__(self):
        enc_string = "Node type {}, Node id {}\n".format(self.__vertex_type, self._id)
        return enc_string + str(self.encoding)

class ConvNode(Node):
    """
    Node to simulate convolutional layer

    Model properties are stored in the __encoding list with the following data
    contained in the correponding indexes.
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
    
    Arguments
        random_node -- if True creates randomises all hyper-parameters.
            If False only filter number, kernel size, stride and padding is randomised.
            (Default = False)
    """

    __vertex_type = "conv"

    def __init__(self, random_node=False):
        super().__init__()
        self._random_conv_filter_num()
        self._random_conv_kernel()
        self._random_conv_stride()
        self.encoding[3] = 'relu'
        self._random_conv_layer_padding()
        self.last_mutate_index = None
        self.pre_mutate_value = None

        if random_node:
            self._random_conv_dropout()
            self._random_pooling_type()
            self._random_pooling_size()
            self._random_pool_stride()
            self._toggle_batch_normalisation()

    def build(self, model):
        """ Builds appropriate keras tensoors for convolutional node. """
        # Check dimension output
        if self.compute_output_dimension((model.get_shape()[1], model.get_shape()[2])) < 1:
            raise DimensionException("Error with dimensions; model shape {}", model.get_shape())
        model = Conv2D(self.encoding[0], self.encoding[2], strides=self.encoding[1], padding=self.encoding[4])(model)
        if self.encoding[9] == 1:           # Batch normalisation layer
            model = BatchNormalization()(model)
        model = Activation(self.encoding[3])(model)
        if self.encoding[6] > 0:            # Pooling layer
            if self.encoding[6] == 1:           # max pooling
                MaxPooling2D((self.encoding[7], self.encoding[7]), strides=self.encoding[8])(model)
            else:                               # average pooling
                AveragePooling2D((self.encoding[7], self.encoding[7]), strides=self.encoding[8])(model)
        if self.encoding[5] > 0:            # Dropout layer
            model = Dropout(self.encoding[8])(model)
        self._logger.info("Dimensions after build of node %d are {}", self.id)
        self._logger.info(model.get_shape())
        self._model = model

    def compute_output_dimension(self, input_dimensions):
        """Computes expected conv node output based on encoding"""
        width, heigth = input_dimensions
        if width != heigth:
            raise ValueError("width and height are not the same")
        if self.encoding[4] == "same":
            output_of_conv = int(width)/self.encoding[1]
        else:
            output_of_conv = int(int(width - self.encoding[2]) / int(self.encoding[1]))
        self._logger.info("output dimensions of convolutional layer for node {0} is {1}".format(str(self.id), 
                                                                                                str(output_of_conv)))
        if self.encoding[6] > 0:
            output_dimension = int((output_of_conv - self.encoding[7]) / self.encoding[8] + 1)
        else:
            output_dimension = output_of_conv
        self._logger.info("output dimensions of pooling layer for node {0} is {1}".format(str(self.id), str(
            output_dimension)))
        return output_dimension

    def mutate(self):
        """Chooses random node hyper-parameters and mutates. """
        rand = random.randrange(0, 9)
        if rand == 0:
            self._random_conv_filter_num()
        elif rand == 1:
            self._random_conv_kernel()
        elif rand == 2:
            self._random_conv_stride()
        elif rand == 3:
            self._random_conv_layer_padding()
        elif rand == 4:
            self._random_conv_dropout()
        elif rand == 5:
            self._random_pooling_type()
        elif rand == 6:
            self._random_pooling_size()
        elif rand == 7:
            self._random_pool_stride()
        else:
            self._toggle_batch_normalisation()

    def reshuffle_dimensions(self):
        """ Performs mutations that adjust node output dimensions. """
        self._random_conv_kernel()
        self._random_conv_stride()

    def _random_conv_filter_num(self):
        min_value, max_value, interval = self.config_min_max_interval('convolutional.layer.filter')
        self.encoding[0] = 2 ** random.randrange(min_value, max_value + 1, interval)  # sets layer units
        self._logger.info("set con filters to %d on node %d", self.encoding[0], self.id)

    def _random_conv_kernel(self):
        self.last_mutate_index = 1
        self.pre_mutate_value = self.encoding[1]
        min_value, max_value, interval = self.config_min_max_interval('convolutional.layer.kernel')
        self.encoding[1] = random.randrange(min_value, max_value + 1, interval)
        self._logger.info("set kernel size to %d on node %d", self.encoding[1], self.id)

    def _random_conv_stride(self):
        self.last_mutate_index = 2
        self.pre_mutate_value = self.encoding[2]
        min_value, max_value, interval = self.config_min_max_interval('convolutional.layer.stride')
        self.encoding[2] = random.randrange(min_value, self.encoding[1] + 1, interval)
        self._logger.info("set conv stride to %d on node %d", self.encoding[2], self.id)

    def _random_conv_layer_padding(self):
        # padding_index = random.randrange(0, 2)
        # if padding_index == 0:
        #     self.encoding[4] = 'same'
        # else:
        #     self.encoding[4] = 'valid'
        self.encoding[4] = 'same'
        self._logger.info("set padding to %s on node %d", self.encoding[4], self.id)

    def _random_conv_dropout(self):
        min_value, max_value, interval = self.config_min_max_interval('conv.layer.dropout')
        self.encoding[5] = (random.randrange(min_value, max_value + 1, interval)) / 10  # Set dropout probability
        self._logger.info("set droupout to %f on node %d", self.encoding[7], self.id)

    def _random_pooling_type(self):
        self.last_mutate_index = 7
        self.pre_mutate_value = self.encoding[7]
        min_value, max_value, interval = self.config_min_max_interval('pooling.type')
        self.encoding[6] = random.randrange(min_value, max_value + 1, interval)
        self._logger.info("set pooling type to %s on node %d", self.encoding[6], self.id)

    def _random_pooling_size(self):
        self.last_mutate_index = 7
        self.pre_mutate_value = self.encoding[7]
        min_value, max_value, interval = self.config_min_max_interval('pooling.filter')
        self.encoding[7] = random.randrange(min_value, max_value + 1, interval)
        self._logger.info("set pooling size to %d on node %d", self.encoding[7], self.id)

    def _random_pool_stride(self):
        self.last_mutate_index = 8
        self.pre_mutate_value = self.encoding[8]
        min_value, max_value, interval = self.config_min_max_interval('convolutional.layer.pool.stride')
        self.encoding[8] = random.randrange(min_value, self.encoding[6] + 1, interval)
        self._logger.info("set pool stride to %d on node %d", self.encoding[8], self.id)

    def _toggle_batch_normalisation(self):
        self.last_mutate_index = 9
        self.pre_mutate_value = self.encoding[9]
        if self.encoding[9] == 1:
            self.encoding[9] = 0
        else:
            self.encoding[9] = 1
        self._logger.info("set batch normalisation to %d on node %d", self.encoding[10], self.id)

    def _undo_last_mutate(self):
        """Returns node to state before last mutation."""
        self.encoding[self.last_mutate_index] = self.pre_mutate_value


class DenseNode(Node):
    """
    Node to simulate dense layer

    Model properties are stored in the __encoding list with the following data
    contained in the correponding indexes.
       0   layer units
       1   dropout
       2   activation
    Arguments:
        random_node -- If true randomies dropout, else dropout is 0. (Default = False)
    """
    __vertex_type = "dense"

    def __init__(self, random_node=False):
        super().__init__()
        self._random_dense_units()
        self.encoding[2] = 'relu'
        if random_node:
            self._random_dense_dropout()

    def build(self, model, output_layer=False, classes=None):
        """
        Builds and returns keras tensor
        
        Arguments:
            model -- tensor from input connections
            output_layer -- If true indicates node is output.
                Activation is softmax, num of perceptrons = classes, dropout
                is excluded. (Default = False).
        """
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
        """Chooses random node hyper-parameters and mutates"""
        rand = random.randrange(0, 2)
        if rand == 0:
            self._random_dense_units()
        else:
            self._random_dense_dropout()

    def _random_dense_units(self):
        min_value, max_value, interval = self.config_min_max_interval('dense.layer.units')
        self.encoding[0] = 2 ** (random.randrange(min_value, max_value + 1, interval))     # Set dense units
        self._logger.info("set dense units to %d on node %d", self.encoding[0], self.id)

    def _random_dense_dropout(self):
        min_value, max_value, interval = self.config_min_max_interval('dense.layer.dropout')
        self.encoding[1] = (random.randrange(min_value, max_value + 1, interval)) / 10  # Set dropout probability
        self._logger.info("set droupout to %f on node %d", self.encoding[1], self.id)


class ConvInputNode(Node):
    """
        Input node of feature extraction graph.
        
        Arguments:
            shape -- shape of input data e.g. (28, 28, 1)
    """
    __vertex_type = "ConvInputNode"

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def build(self):
        """Builds and stores Keras tensor."""
        self._model = Input(self.shape)

    @property
    def output_dimension(self):
        return tuple(self.shape)


class ConvOutputNode(Node):
    """Input node of feature extraction graph."""
    
    __vertex_type = "ConvOutputNode"

    def __init__(self):
        super().__init__()

    def build(self, model):
        """ Builds and stoers Keras model. """
        self._model = Flatten()(model)

    @property
    def output_dimension(self):
        return self._model.get_shape()[1]

