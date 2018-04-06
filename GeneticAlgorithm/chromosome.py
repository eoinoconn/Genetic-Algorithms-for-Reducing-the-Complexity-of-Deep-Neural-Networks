from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, AveragePooling2D, \
    concatenate, Input, BatchNormalization, Activation, Concatenate
from keras.models import Model
import configparser
import logging
import random
import csv


class GeneticObject(object):

    @staticmethod
    def config_min_max_interval(config_name):
        config = configparser.ConfigParser()
        config.read(
            "C:/Users/eoino/Documents/GitHub/Genetic-Algorithms-for-Reducing-the-Complexity-of-Deep-Neural-Networks"
            "/GeneticAlgorithm/Config/training_parameters.ini")
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
        self.__id = Node._id
        self.active = True
        self._logger = logging.getLogger('geneset')
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
        self.tensor = None

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

    def build(self):
        model = Conv2D(self.encoding[0], self.encoding[2], strides=self.encoding[1], padding=self.encoding[4])
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
        self.tensor = model

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

    def random_conv_filter_num(self):
        min_value, max_value, interval = self.config_min_max_interval('convolutional.layer.filter')
        self.encoding[0] = 2 ** random.randrange(min_value, max_value + 1, interval)  # sets layer units
        self._logger.info("set con filters to %d on node %d", self.encoding[0], self.id)

    def random_conv_kernel(self):
        min_value, max_value, interval = self.config_min_max_interval('convolutional.layer.kernel')
        self.encoding[1] = random.randrange(min_value, max_value + 1, interval)
        self._logger.info("set kernel size to %d on node %d", self.encoding[1], self.id)

    def random_conv_stride(self):
        min_value, max_value, interval = self.config_min_max_interval('convolutional.layer.stride')
        self.encoding[2] = random.randrange(min_value, self.encoding[1] + 1, interval)
        self._logger.info("set conv stride to %d on node %d", self.encoding[2], self.id)

    def random_conv_layer_padding(self):
        padding_index = random.randrange(0, 2)
        if padding_index == 0:
            self.encoding[4] = 'same'
        else:
            self.encoding[4] = 'valid'
        self._logger.info("set padding to %s on node %d", self.encoding[4], self.id)

    def random_conv_dropout(self):
        min_value, max_value, interval = self.config_min_max_interval('conv.layer.dropout')
        self.encoding[5] = (random.randrange(min_value, max_value + 1, interval)) / 10  # Set dropout probability
        self._logger.info("set droupout to %f on node %d", self.encoding[7], self.id)

    def random_pooling_type(self):
        min_value, max_value, interval = self.config_min_max_interval('pooling.type')
        self.encoding[6] = random.randrange(min_value, max_value + 1, interval)
        self._logger.info("set pooling type to %s on node %d", self.encoding[6], self.id)

    def random_pooling_size(self):
        min_value, max_value, interval = self.config_min_max_interval('pooling.filter')
        self.encoding[7] = random.randrange(min_value, max_value + 1, interval)
        self._logger.info("set pooling size to %d on node %d", self.encoding[7], self.id)

    def random_pool_stride(self):
        min_value, max_value, interval = self.config_min_max_interval('convolutional.layer.pool.stride')
        self.encoding[8] = random.randrange(min_value, self.encoding[6] + 1, interval)
        self._logger.info("set pool stride to %d on node %d", self.encoding[8], self.id)

    def toggle_batch_normalisation(self):
        if self.encoding[9] == 1:
            self.encoding[9] = 0
        else:
            self.encoding[9] = 1
        self._logger.info("set batch normalisation to %d on node %d", self.encoding[10], self.id)


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
        self.tensor = None

    def build(self):
        self.tensor = Input(self.shape)
        return self.tensor


# class FlattenNode(Node):
#    def __init__(self):
#        super().__init__()
#        self._vertex_type = "flatten"
#
#    @staticmethod
#    def build(model):
#        return Flatten()(model)


class ConvOutputNode(Node):

    def __init__(self):
        super().__init__()
        self._vertex_type = "output"
        self.tensor = None

    def build(self):
        self.tensor = Flatten()


class Chromosome(GeneticObject):

    _id = 1

    def __init__(self):
        self.id = Chromosome._id
        self.hyperparameters = [0 for x in range(0, 25)]
        self.fitness = None
        self.accuracy = None
        self.parameters = None
        self.input_conv_id = 0
        self.output_conv_id = 0
        self.age = 0
        self.vertices = {}
        self.conv_nodes = []
        self.dense_nodes = []
        self.shape = (32, 32, 3)
        Chromosome._id += 1

        config = configparser.ConfigParser()
        config.read('GeneticAlgorithm/Config/training_parameters.ini')
        logger = logging.getLogger('Chromosome')
        logger.info("creating parent genes")

        self.hyperparameters[0] = 'categorical_crossentropy'    # loss
        self.hyperparameters[1] = 'adam'                        # optimizer
        self.random_batch_size()

        # Build minimal structure
        self.minimal_structure()

    def minimal_structure(self):
        self.add_node(ConvOutputNode())
        self.output_conv_id = self.conv_nodes[0].id
        self.add_node(DenseNode())
        self.input_conv_id = self.conv_nodes[2].id
        self.add_node(ConvInputNode(self.shape))

    def add_node(self, node):
        if isinstance(node, ConvNode):
            self.conv_nodes.append(node)
            self.vertices[node.id] = []
        elif isinstance(node, DenseNode):
            self.dense_nodes.append(node)
        else:
            raise TypeError("node should be of type ConvNode or DenseNode")

    # add id of the node that inputs to node
    def add_vertex(self, node, input_node_id):
        self.vertices[node.id].append(input_node_id)

    def remove_node(self, node_to_remove):
        if isinstance(node_to_remove, Node):
            node_to_remove = node_to_remove.id
        for node in self.conv_nodes:
            if node.id == node_to_remove:
                node.active = False
                return
        for node in self.dense_nodes:
            if node.id == node_to_remove:
                node.active = False
                return

    def remove_vertex(self, node, input_node_id):
        if isinstance(input_node_id, Node):
            input_node_id = input_node_id.id
        if isinstance(node, Node):
            node = node.id
        self.vertices[node].remove(input_node_id)

    def build(self):
        self.build_conv_tensors()
        self.assemble_graph()
        model = self.recurrently_build_list(self.conv_nodes[self.output_conv_id].tensor, self.dense_nodes, 0)
        input_layer = self.conv_nodes[self.input_conv_id].tensor
        return Model(inputs=input_layer, outputs=model)

    def build_conv_tensors(self):
        for node in self.conv_nodes:
                node.build()

    def assemble_graph(self):
        # for each node starting from output node, we add to the model
        for node in self.conv_nodes:
            tensor = node.tensor
            id_of_input_nodes = self.vertices[node.id]
            tensors_to_concatenate = []
            for id in id_of_input_nodes:
                tensors_to_concatenate.append(self.conv_nodes[id].tensor)
            if len(tensors_to_concatenate) > 0:
                tensor = tensor(Concatenate(tensors_to_concatenate))
                node.tensor = tensor

    def recurrently_build_list(self, model, dense_nodes, index):
        if index < (len(dense_nodes) - 1):
            model = dense_nodes[index].build(model)
            return self.recurrently_build_list(model, dense_nodes, (index + 1))
        else:
            return dense_nodes[index].build(model, output_layer=True, classes=10)

    def mutate(self):
        rand = random.randrange(0, 4)
        if rand == 0:
            """ Add node"""
            rand = random.randrange(0, 2)
            if rand == 0:
                input_node = self.random_conv_node()
                output_node = self.random_conv_node()
                new_node = ConvNode(random_node=True)
                self.add_vertex(input_node, new_node)
                self.add_node(new_node)
                self.add_vertex(new_node, output_node)
            else:
                self.add_node(DenseNode(random_node=True))
        elif rand == 1:
            """Add edge"""
            rand_node_1 = self.random_conv_node()
            rand_node_2 = self.random_conv_node()
            self.vertices[rand_node_1].append(rand_node_2)
        elif rand == 2:
            """ Mutate hyperparameters"""
            self.random_batch_size()
        else:
            """Mutate random node"""
            rand = random.randrange(0,2)
            if rand == 0:
                self.random_conv_node().mutate()
            else:
                self.random_dense_node().mutate()

    def random_conv_node(self):
        return random.choice(self.conv_nodes)

    def random_dense_node(self):
        return random.choice(self.dense_nodes)

    def random_batch_size(self):
        min_value, max_value, interval = self.config_min_max_interval('chromosome.batchsize')
        self.hyperparameters[2] = random.randrange(min_value, max_value + 1, interval)  # batch size

    def log(self, generation_num):
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
        return len(self.conv_nodes)

    def __str__(self):
        string = ""
        for obj in self.conv_nodes:
            string += str(obj)
        string += str(self.vertices)
        return string
