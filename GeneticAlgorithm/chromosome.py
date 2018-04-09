from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, AveragePooling2D, \
    concatenate, Input, BatchNormalization, Activation, Concatenate
from keras.models import Model
from GeneticAlgorithm.node import *
from GeneticAlgorithm.fitness import assess_chromosome_fitness
import configparser
import logging
import random
import csv


class Chromosome(GeneticObject):

    _id = 1

    def __init__(self, input_shape):
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
        self.shape = input_shape
        Chromosome._id += 1

        config = configparser.ConfigParser()
        config.read('C:/Users/eoino/Documents/GitHub/Genetic-Algorithms-for-Reducing-the-Complexity-of-Deep-Neural'
                    '-Networks/GeneticAlgorithm/Config/training_parameters.ini')
        logger = logging.getLogger('Chromosome')
        logger.info("creating parent genes")

        self.hyperparameters[0] = 'categorical_crossentropy'    # loss
        self.hyperparameters[1] = 'adam'                        # optimizer
        self.random_batch_size()

        # Build minimal structure
        if config['initial.generation'].getboolean('random_initial_generation'):
            self.random_initial_generation()
        else:
            self.minimal_structure()

    def random_initial_generation(self):
        self.minimal_structure()

    def minimal_structure(self):
        self.add_node(ConvOutputNode())
        self.output_conv_id = self.conv_nodes[0].id
        self.add_node(ConvInputNode(self.shape))
        self.input_conv_id = self.conv_nodes[1].id
        self.add_vertex(self.output_conv_id, self.input_conv_id)
        self.add_node(DenseNode())

    def add_node(self, node):
        if isinstance(node, (ConvNode, ConvOutputNode, ConvInputNode)):
            self.conv_nodes.append(node)
            self.vertices[node.id] = []
        elif isinstance(node, DenseNode):
            self.dense_nodes.append(node)
        else:
            raise TypeError("node should be of type ConvNode or DenseNode")

    # add id of the node that inputs to node
    def add_vertex(self, node, input_node_id):
        if isinstance(node, Node):
            node = node.id
        self.vertices[node].append(input_node_id)

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
        self.recurrently_build_graph(self.input_conv_id)
        model = self.recurrently_build_list(self.conv_by_id(self.output_conv_id).model, self.dense_nodes, 0)
        input_layer = self.conv_by_id(self.input_conv_id).model
        return Model(inputs=input_layer, outputs=model)

    """Each conv node needs to know its input and output dimensions. But this should only be calculated when building. 
    Can only be calculated when building. Both should be stored as instance variables. Both should be swiped upon 
    completion of the build, along with the tensors """

    def recurrently_build_graph(self, id):
        node = self.conv_by_id(id)
        input_node_ids = self.vertices[id]
        if len(input_node_ids) > 0:
            smallest_dimension = self.find_smallest_dimension(input_node_ids)
            tensors_to_concatenate = []
            for input_id in input_node_ids:
                input_node = self.conv_by_id(input_id)
                if not input_node.is_built():
                    return
                tensors_to_concatenate.append(self.downsample_to(input_node.model, smallest_dimension, input_node.output_dimension))
            if len(tensors_to_concatenate) > 1:
                node.build(Concatenate(axis=3)(tensors_to_concatenate))
            elif len(tensors_to_concatenate) == 1:
                node.build(tensors_to_concatenate[0])
            node.compute_output_dimension(smallest_dimension)
        else:
            node.build()
        for output_id in self.conv_node_outputs(id):
            self.recurrently_build_graph(output_id)

    def downsample_to(self, tensor, downsample_to, input_at):
        stride = 1
        while True:
            if downsample_to < int(input_at/(stride + 1)):     # could be a problem area
                stride += 1
            else:
                kernel = (int(input_at/(stride + 1)) - downsample_to)
                break
        return MaxPooling2D(kernel, strides=stride)(tensor)

    def find_smallest_dimension(self, input_node_ids):
        if len(input_node_ids) == 0:
            return
        current_smallest = 1000
        for id in input_node_ids:
            if self.conv_by_id(id).output_dimension < current_smallest:
                current_smallest = self.conv_by_id(id).output_dimension
        return current_smallest

    def evaluate(self):
        raise NotImplementedError
        # assess_chromosome_fitness()

    def recurrently_build_list(self, model, dense_nodes, index):
        if index < (len(dense_nodes) - 1):
            model = dense_nodes[index].build(model)
            return self.recurrently_build_list(model, dense_nodes, (index + 1))
        elif index == (len(dense_nodes) - 1):
            return dense_nodes[index].build(model, output_layer=True, classes=10)
        else:
            return model

    def mutate(self):
        rand = random.randrange(0, 4)
        if rand == 0:
            """ Add node"""
            rand = random.randrange(0, 2)
            if rand == 0:
                self.add_random_conv_node()
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

    def add_random_conv_node(self):
        new_node = ConvNode(random_node=True)
        self.add_node(new_node)
        new_node_id = new_node.id
        while True:
            input_node_id = self.random_conv_node().id
            if input_node_id == self.output_conv_id:
                continue
            output_node_id = self.random_conv_node().id
            if output_node_id == self.input_conv_id:
                continue
            self.add_vertex(new_node, input_node_id)
            self.add_vertex(output_node_id, new_node.id)
            if not self.creates_cycle():
                break
            self.remove_vertex(new_node_id, input_node_id)
            self.remove_vertex(output_node_id, new_node_id)

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

    def conv_nodes_iterator(self):
        for node in self.conv_nodes:
            yield node

    def conv_by_id(self, id):
        for node in self.conv_nodes_iterator():
            if node.id == id:
                return node

    def conv_node_outputs(self, id):
        for key, contents in self.vertices.items():
            if id in contents:
                yield key

    def increment_age(self):
        self.age += 1

    def mash(self):
        raise NotImplementedError

    def creates_cycle(self):
        """Return True if the directed graph has a cycle.
        The graph must be represented as a dictionary mapping vertices to
        iterables of neighbouring vertices. For example:

        """
        visited = set()
        path = [object()]
        path_set = set(path)
        stack = [iter(self.vertices)]
        while stack:
            for v in stack[-1]:
                if v in path_set:
                    return True
                elif v not in visited:
                    visited.add(v)
                    path.append(v)
                    path_set.add(v)
                    stack.append(iter(self.vertices.get(v, ())))
                    break
            else:
                path_set.remove(path.pop())
                stack.pop()
        return False

    def __len__(self):
        return len(self.conv_nodes)

    def __str__(self):
        string = ""
        for obj in self.conv_nodes:
            string += str(obj)
        string += str(self.vertices)
        return string
