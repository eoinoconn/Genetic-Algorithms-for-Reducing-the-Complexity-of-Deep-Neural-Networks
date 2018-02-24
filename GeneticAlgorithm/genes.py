import logging
import copy

from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, AveragePooling2D, concatenate, Input
from keras.models import Sequential, Model
from keras.utils import print_summary

from GeneticAlgorithm.fitness import assess_chromosome_fitness

MAX_LAYERS = 50
LAYER_DEPTH = 8
CLASSES = 10


class LoggerMixin:
    @property
    def logger(self):
        # component = "{}.{}".format(type(self).__module__, type(self).__name__)
        return logging.getLogger('genes')

    def log_geneset(self, log_file='geneset'):
        logger = logging.getLogger(log_file)
        logger.info("Geneset id: %d, age: %d", self.id, self.age)
        logger.info(self.__str__() + "\n")

    def log_best(self, fitness, accuracy, parameters, log_file='resultMetrics'):
        self.log_geneset()
        logger = logging.getLogger(log_file)
        logger.info("new best chromosome, id = %d, age = %d", self.id, self.age)
        print_summary(self.build_model(), print_fn=logger.info)
        logger.info("Fitness: %.6f\tAccuracy: %.6f\tParameters %d\n", fitness, accuracy, parameters)


class ModelMixin:

    def build_model(self):
        input_layer = Input(shape=self.input_shape)
        model = input_layer
        for x in range(self.__len__() + 1):
            # check if output layer, hidden layer or no layer at all
            if (self.genes[x][0] != 0) and (self.genes[x + 1][0] == 0):  # Check if output layer
                model = self.build_layer(model, self.genes[x], output_layer=True)
            elif self.genes[x][0] != 0:  # else check if not empty layer
                model = self.build_layer(model, self.genes[x])
            else:
                return Model(inputs=input_layer, outputs=model)

    def build_layer(self, model, layer, output_layer=False):
        if layer[0] == 1:  # Dense Layer
            if output_layer:  # output layer
                return Dense(CLASSES, activation='softmax')(model)
            else:  # hidden layer
                model = Dense(layer[1], activation='relu')(model)
                if layer[3] > 0:
                    model = Dropout(layer[3])(model)
                return model

        elif layer[0] == 2:  # conv layer
            # hidden layer
            model = Conv2D(layer[1], (layer[3], layer[3]), activation='relu')(model)
            if layer[5] > 0:  # check for pooling layer
                model = self.pooling_layer(model, layer)
            return model

        elif layer[0] == 3:  # Flatten layer
            return Flatten()(model)

        elif layer[0] == 4:
            return self.inception_module(model)

        else:
            raise NotImplementedError('Layers not yet implemented')

    def pooling_layer(self, input_layer, layer):
        if layer[5] == 1:  # max pooling
            return MaxPooling2D((layer[6], layer[6]))(input_layer)
        else:
            return AveragePooling2D((layer[6], layer[6]))(input_layer)

    def model_summary(self):
        self.build_model().summary()

    def inception_module(self, input_layer):
        tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_layer)
        tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)
        tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_layer)
        tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)
        tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_layer)
        tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)
        return concatenate([tower_1, tower_2, tower_3], axis=3)


class Genes(LoggerMixin, ModelMixin):
    ids = 0

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

    def increment_age(self):
        self.age += 1

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
        self.log_best(assess_chromosome_fitness(self, evaluate_best=evaluate_best,
                                                log_csv=log_csv, **training_data))

    def mash(self):
        mash = copy.deepcopy(self.genes)
        mash.append(self.hyperparameters)
        sum = 0
        for layer in mash:
            sum += hash(frozenset(layer))
        return hash(sum)

    def assume_values(self, values):
        self.fitness = values[0]
        self.accuracy = values[1]
        self.parameters = values[2]

    def __str__(self):
        str = self.hyperparameters.__str__() + "\n"
        for layer in self.iterate_layers():
            str += layer.__str__() + "\n"
        return str

    def __len__(self):
        for x in range(0, MAX_LAYERS):
            if self.genes[x][0] == 0:
                return x

    def __gt__(self, other):
        return self.fitness > other.fitness
