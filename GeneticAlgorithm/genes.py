import logging
import copy

from keras.utils import print_summary

from GeneticAlgorithm.fitness import assess_chromosome_fitness
from GeneticAlgorithm.chromosome_model import ChromosomeModel

MAX_LAYERS = 50
LAYER_DEPTH = 14


class Genes(object):
    ids = 1

    def __init__(self, input_shape):
        self.genes = [[0 for x in range(0, LAYER_DEPTH)] for y in range(0, MAX_LAYERS)]
        self.hyperparameters = [0 for x in range(0, 25)]
        self.fitness = None
        self.accuracy = None
        self.parameters = None
        self.input_shape = input_shape
        self.id = Genes.ids
        self.age = 0
        Genes.ids += 1

    def set_hyperparameters(self, new_hyperparameters):
        self.hyperparameters = new_hyperparameters

    def add_layer(self, layer, index=None):
        if index is None:
            self.genes[self.__len__()] = layer
        else:
            genes_length = self.__len__()
            for i in range(index, genes_length + 1):
                temp_layer = self.genes[i]
                self.genes[i] = layer
                layer = temp_layer

    def overwrite_layer(self, layer, index):
        self.genes[index] = layer

    def remove_layer(self, index=None):
        if index is None:
            self.genes[self.__len__()] = [0 for x in range(0, LAYER_DEPTH)]
        else:
            genes_length = self.__len__()
            for i in range(index, genes_length):
                self.genes[i] = self.genes[i + 1]

    def get_layer(self, index):
        return self.genes[index]

    def get_layer_type(self, index):
        return self.get_layer(index)[0]

    def save_layer_weights(self, index, weights_and_biases):
        layer = self.get_layer(index)
        layer[-1] = weights_and_biases
        self.overwrite_layer(layer, index)

    def save_batch_layer_weights(self, index, weights_and_biases):
        layer = self.get_layer(index)
        layer[-2] = weights_and_biases
        self.overwrite_layer(layer, index)

    def get_layer_weights(self, index):
        return self.get_layer(index)[-1]

    def num_dense_layers(self):
        count = 0
        for layer in self.iterate_layers():
            if layer[0] == 1:
                count += 1
        return count

    def num_conv_layers(self):
        count = 0
        for layer in self.iterate_layers():
            if layer[0] == 2:
                count += 1
        return count

    def num_incep_layers(self):
        count = 0
        for layer in self.iterate_layers():
            if layer[0] == 4:
                count += 1
        return count

    def iterate_layers(self):
        for x in range(0, self.__len__()):
            layer = self.get_layer(x)
            yield layer

    def increment_age(self):
        self.age += 1

    def find_flatten(self):
        count = 0
        for layer in self.iterate_layers():
            if layer[0] == 3:
                return count
            count += 1
        return -1

    def clear_genes(self):
        for i in range(0, self.__len__()):
            self.overwrite_layer([0 for x in range(0, LAYER_DEPTH)], i)

    def assess_fitness(self, training_data, log_csv=False):
        self.fitness, self.accuracy, self.parameters = assess_chromosome_fitness(self,
                                                                                 log_csv=log_csv, **training_data)

    def assess_fitness_with_test(self, training_data, log_csv=False):
        evaluate_best = True
        self.fitness, self.accuracy, self.parameters = assess_chromosome_fitness(self, evaluate_best=evaluate_best,
                                                                                 log_csv=log_csv, **training_data)

    def log_geneset(self, log_file='geneset'):
        logger = logging.getLogger(log_file)
        logger.info("Geneset id: %d, age: %d", self.id, self.age)
        logger.info(self.__str__() + "\n")

    def log_best(self, log_file='resultMetrics'):
        self.log_geneset()
        logger = logging.getLogger(log_file)
        logger.info("new best chromosome, id = %d, age = %d", self.id, self.age)
        print_summary(self.build_model(), print_fn=logger.info)
        logger.info("Fitness: %.6f\tAccuracy: %.6f\tParameters %d\n", self.fitness, self.accuracy, self.parameters)

    def build_model(self):
        model = ChromosomeModel(self.genes, 10, self.input_shape, self.__len__())
        return model.build_model()

    @property
    def logger(self):
        return logging.getLogger('genes')

    def mash(self):
        mash = copy.deepcopy(self.genes)
        mash.append(self.hyperparameters)
        sum = 0
        for layer in mash:
            sum += hash(frozenset(layer[:-2]))
        return hash(sum)

    def assume_values(self, values):
        self.fitness = values[0]
        self.accuracy = values[1]
        self.parameters = values[2]

    def __str__(self):
        str = self.hyperparameters.__str__() + "\n"
        for layer in self.iterate_layers():
            if layer[-1] is not 0:
                shape1 = layer[-1].shape
            if layer[-2] is not 0:
                shape2 = layer[-1].shape
            str += layer[:-2].__str__() + shape1 + shape2 + "\n"
        return str

    def __len__(self):
        for x in range(0, MAX_LAYERS):
            if self.genes[x][0] == 0:
                return x

    def __gt__(self, other):
        return self.fitness > other.fitness
