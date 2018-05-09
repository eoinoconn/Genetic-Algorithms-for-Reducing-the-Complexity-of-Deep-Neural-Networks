import random
import configparser
import logging
import csv
from keras.layers import Concatenate, MaxPooling2D
from keras.backend import clear_session
from keras.models import Model
from tensorflow import reset_default_graph
from GeneticAlgorithm.node import GeneticObject, Node, ConvNode, DenseNode, ConvInputNode, ConvOutputNode
from GeneticAlgorithm.exceptions import *
from GeneticAlgorithm.fitness import Fitness


class Chromosome(GeneticObject):

    _id = 1

    def __init__(self, input_shape):
        self.id = Chromosome._id
        self.hyperparameters = [0 for x in range(0, 25)]
        self.fitness = None
        self.accuracy = None
        self.parameters = None
        self._logger = None
        self.input_conv_id = 0
        self.output_conv_id = 0
        self.age = 0
        self.vertices = {}
        self.conv_nodes = []
        self.dense_nodes = []
        self.shape = input_shape
        Chromosome._id += 1

        config = configparser.ConfigParser()
        config.read("../training_parameters.ini")
        self._logger = logging.getLogger('Chromosome')
        self._logger.info("creating parent genes")

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
                for node_id in self.conv_node_outputs(node_to_remove):
                    self.remove_vertex(node_id, node_to_remove)
                self.vertices.pop(node_to_remove)
                return
        for idx, val in enumerate(self.dense_nodes):
            if val.id == node_to_remove:
                node_to_remove = idx
        self.dense_nodes.pop(node_to_remove)

    def remove_vertex(self, node, input_node_id):
        if isinstance(input_node_id, Node):
            input_node_id = input_node_id.id
        if isinstance(node, Node):
            node = node.id
        self.vertices[node].remove(input_node_id)

    def build(self):
        self._logger.info(str(self))
        self.recurrently_build_graph(self.input_conv_id)
        model = self.recurrently_build_list(self.conv_by_id(self.output_conv_id).model, self.dense_nodes, 0)
        input_layer = self.conv_by_id(self.input_conv_id).model
        return Model(inputs=input_layer, outputs=model)

    def destroy_models(self):
        for node in self.conv_nodes_iterator():
            node.delete_model()
        reset_default_graph()  # for being sure
        clear_session()  # removing session, it will instance another

    def recurrently_build_graph(self, conv_id):
        """
        Build the graph of convolutional layers and assemble tensors width first.
        :param conv_id:
        :return:
        """
        node = self.conv_by_id(conv_id)
        if not node.active:
            return
        input_node_ids = list(set(self.vertices[conv_id]))
        if len(input_node_ids) > 0:
            if not self.check_inputs_built(input_node_ids):
                return
            smallest_dimension = self.find_smallest_dimension(input_node_ids)
            tensors_to_concatenate = []
            for input_id in input_node_ids:
                input_node = self.conv_by_id(input_id)
                if not input_node.active:
                    continue
                tensors_to_concatenate.append(self.downsample_to(input_node.model, smallest_dimension,
                                                                 input_node.output_dimension))
            if len(tensors_to_concatenate) > 1:
                node.build(Concatenate(axis=-1)(tensors_to_concatenate))
            elif len(tensors_to_concatenate) == 1:
                node.build(tensors_to_concatenate[0])
        else:
            node.build()
        for output_id in self.conv_node_outputs(conv_id):
            self.recurrently_build_graph(output_id)

    def check_inputs_built(self, input_node_ids):
        for input_id in input_node_ids:
            input_node = self.conv_by_id(input_id)
            if not input_node.active:
                continue
            if not input_node.is_built():
                return False
        return True

    def downsample_to(self, tensor, downsample_to, input_at):
        self._logger.info("Downsampling for node %d from input at %d to %d", self.id, input_at, downsample_to)
        stride = 1
        if downsample_to == input_at:
            return tensor
        kernel = 1
        while True:
            if int((input_at - kernel) / (stride + 1) + 1) == downsample_to:  # could be a problem area
                stride += 1
                break
            elif int((input_at - kernel) / (stride + 1) + 1) > downsample_to:  # could be a problem area
                stride += 1
            else:
                kernel = -stride * (downsample_to - 1) + input_at
                break
        tensor = MaxPooling2D(kernel, strides=stride)(tensor)
        self._logger.info("Output of downsampling: " + str(tensor.get_shape()))
        return tensor

    def find_smallest_dimension(self, input_node_ids):
        """Find smallest output dimensions in the given list of nodes"""
        self._logger.info("Finding smallest dimension for node %d, with inputs" + str(input_node_ids))
        if len(input_node_ids) == 1:
            self._logger.info("Only one input for node %d, of type %s",
                              input_node_ids[0],
                              self.conv_by_id(input_node_ids[0]).vertex_type)
            return self.conv_by_id(input_node_ids[0]).output_dimension
        current_smallest = 1000
        for ids in input_node_ids:
            if not self.conv_by_id(ids).active:
                continue
            if self.conv_by_id(ids).output_dimension < current_smallest:
                current_smallest = self.conv_by_id(ids).output_dimension
        self._logger.info("Smallest dimension is %d", current_smallest)
        return current_smallest

    def evaluate(self, training_data):
        self._logger.info("Evaluating fitness of chromosome %d, age %d", self.id, self.age)
        fit = Fitness()
        self.fitness, self.accuracy, self.parameters = fit(self.build(), self.hyperparameters,
                                                                                 **training_data)
        self.destroy_models()

    def recurrently_build_list(self, model, dense_nodes, index):
        if index < (len(dense_nodes) - 1):
            model = dense_nodes[index].build(model)
            return self.recurrently_build_list(model, dense_nodes, (index + 1))
        elif index == (len(dense_nodes) - 1):
            return dense_nodes[index].build(model, output_layer=True, classes=10)
        else:
            return model

    def mutate(self):
        self._logger.info("Mutating chromosome %d", self.id)
        while True:
            rand = random.randrange(0, 3)
            if rand == 0:
                """ Add node"""
                rand = random.randrange(0, 2)
                if rand == 0:
                    try:
                        self._logger.info("Attempting to add conv node...")
                        self.add_random_conv_node()
                    except CantAddNode:
                        self._logger.info("Failed to add conv node...")
                        continue
                    break
                else:
                    self._logger.info("Attempting to add dense node...")
                    self.add_random_dense_node()
            elif rand == 1:
                """Add edge"""
                self._logger.info("Attempting to add edge...")
                self.add_random_vertex()
                break
            elif rand == 2:
                """ Mutate hyperparameters"""
                self._logger.info("Mutating hyperparameters")
                self.random_batch_size()
                break
            else:
                """Mutate random node"""
                rand = random.randrange(0, 2)
                if rand == 0:
                    node_to_mutate = self.random_conv_node()
                else:
                    node_to_mutate = self.random_dense_node()

                self._logger.info("Attempting to mutate node %d", node_to_mutate)
                node_to_mutate.mutate()
                try:
                    self.build()
                except DimensionException:
                    node_to_mutate.undo_last_mutate()
                    self._logger.info("Mutation failed on node %d", node_to_mutate)
                    continue
                break

    def add_random_dense_node(self):
        self.add_node(DenseNode(random_node=True))

    def add_random_vertex(self):
        for i in range(50):
            input_node = self.random_conv_node()
            output_node = self.random_conv_node()
            if (output_node.id == self.input_conv_id) or (input_node.id == self.output_conv_id) or not \
                    (output_node.active or input_node.active) or \
                    (input_node == self.input_conv_id and output_node == self.output_conv_id):
                continue
            self.add_vertex(output_node.id, input_node.id)
            if self.creates_cycle():
                self.remove_vertex(output_node.id, input_node.id)
                continue
            try:
                self.build()
                return True
            except DimensionException:
                self.remove_vertex(output_node.id, input_node.id)
                continue
        return False

    def add_random_conv_node(self):
        input_node_id, output_node_id = self.find_random_vertex()
        new_node = ConvNode(random_node=True)
        self.add_node(new_node)
        new_node_id = new_node.id
        self.remove_vertex(output_node_id, input_node_id)
        self.add_vertex(new_node_id, input_node_id)
        self.add_vertex(output_node_id, new_node_id)
        self._logger.info("Attempting to add conv node with id %d", new_node_id)
        for i in range(100):
            try:
                self.build()
            except DimensionException:
                self.conv_by_id(new_node_id).reshuffle_dimensions()
                continue
            if not self.creates_cycle():
                return
        self.remove_vertex(new_node_id, input_node_id)
        self.remove_vertex(output_node_id, new_node_id)
        self.add_vertex(output_node_id, input_node_id)
        self.remove_node(new_node_id)
        raise CantAddNode

    def find_random_vertex(self):
        while True:
            random_output_id = random.choice(list(self.vertices.keys()))
            if random_output_id == self.input_conv_id:
                continue
            break
        random_input_id = random.choice(self.vertices[random_output_id])
        return random_input_id, random_output_id

    def random_conv_node(self):
        while True:
            node = random.choice(self.conv_nodes)
            if node.active:
                return node

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
                                 ])

    def conv_nodes_iterator(self):
        for node in self.conv_nodes:
            yield node

    def conv_by_id(self, find_id):
        for node in self.conv_nodes_iterator():
            if node.id == find_id:
                return node

    def conv_node_outputs(self, find_id):
        for key, contents in self.vertices.items():
            if find_id in contents:
                yield key

    def num_active_conv_nodes(self):
        count = 0
        for node in self.conv_nodes_iterator():
            if node.active:
                count += 1
        return count

    def increment_age(self):
        self.age += 1

    def creates_cycle(self):
        """Return True if the directed graph has a cycle.
        The graph must be represented as a dictionary mapping vertices to
        iterables of neighbouring vertices. For example:

        """
        
        self._logger.info("Checking for created cycle")
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

    def num_conv_nodes(self):
        return len(self.conv_nodes)

    def num_dense_nodes(self):
        return len(self.dense_nodes)

    def __len__(self):
        return len(self.conv_nodes) + len(self.dense_nodes)

    def __str__(self):
        string = ""
        for obj in self.conv_nodes:
            string += (str(obj) + "\n")
        string += str(self.vertices)
        return string
