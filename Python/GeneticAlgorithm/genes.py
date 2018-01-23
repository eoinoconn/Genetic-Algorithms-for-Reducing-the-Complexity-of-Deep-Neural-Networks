from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Conv1D, MaxPooling2D, Dropout, Flatten, AveragePooling2D
from itertools import count
import logging

from Python.GeneticAlgorithm.fitness import assess_chromosome_fitness

MAX_LAYERS = 50
LAYER_DEPTH = 8
INPUT_SHAPE = (28, 28, 1)
CLASSES = 10


class Genes(object):
    ids = 0

    def __init__(self):
        self.logger = logging.getLogger('genes')
        self.logger.info("initialising genes")
        self.genes = [[0 for x in range(0, LAYER_DEPTH)] for y in range(0, MAX_LAYERS)]
        self.hyperparameters = [0 for x in range(0, 25)]
        self.model = Sequential()
        self.fitness = None
        self.id = Genes.ids
        Genes.ids += 1

    def set_hyperparameters(self, new_hyperparameters):
        self.hyperparameters = new_hyperparameters

    def iterate_id(self):
        self.id += 1

    def add_layer(self, layer, index=None):
        if index is None:
            self.genes[self.__len__()] = layer
        else:
            genes_length = self.__len__()
            for i in range(index, genes_length+1):
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

    def num_dense(self):
        count = 0
        for layer in self.iterate_layers():
            if layer[0] == 1:
                count += 1
        return count

    def num_conv(self):
        count = 0
        for layer in self.iterate_layers():
            if layer[0] == 2:
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

    def build_model(self):
        self.model = Sequential()
        input_layer = True
        for x in range(self.__len__()+1):
            # check if output layer, hidden layer or no layer at all
            if (self.genes[x][0] != 0) and (self.genes[x+1][0] == 0):       # Check if output layer
                self.build_layer(self.genes[x], output_layer=True)
            elif self.genes[x][0] != 0:                                     # else check if not empty layer
                self.build_layer(self.genes[x], input_layer=input_layer)
                input_layer = False
            else:
                return self.model

    def build_layer(self, layer, input_layer=False, output_layer=False):
        if layer[0] == 1:   # Dense Layer
            if input_layer:           # input layer
                self.model.add(Dense(layer[1], input_shape=INPUT_SHAPE, activation='relu'))
                if layer[3] > 0:
                    self.model.add(Dropout(layer[3]))
            elif output_layer:        # output layer
                self.model.add(Dense(CLASSES, activation='softmax'))
            else:               # hidden layer
                self.model.add(Dense(layer[1], activation='relu'))
                if layer[3] > 0:
                    self.model.add(Dropout(layer[3]))

        elif layer[0] == 2:     # conv layer
            if input_layer and output_layer:    # input and output
                self.model.add(Conv2D(CLASSES, (layer[3], layer[3]), input_shape=INPUT_SHAPE, activation='softmax'))
            elif input_layer:           # input layer
                self.model.add(Conv2D(layer[1], (layer[3], layer[3]), input_shape=INPUT_SHAPE, activation='relu'))
                if layer[5] > 0:  # check for pooling layer
                    self.pooling_layer(layer)
            elif output_layer:        # output layer
                self.model.add(Conv2D(CLASSES, (layer[3], layer[3]), activation='softmax'))
            else:               # hidden layer
                self.model.add(Conv2D(layer[1], (layer[3], layer[3]), activation='relu'))
                if layer[5] > 0:    # check for pooling layer
                    self.pooling_layer(layer)
        elif layer[0] == 3:
            if input_layer:
                self.model.add(Flatten())
            else:
                self.model.add(Flatten(input_shape=INPUT_SHAPE))
        else:
            raise NotImplementedError('Layers not yet implemented')

    def pooling_layer(self, layer):
        if layer[5] == 1:   # max pooling
            self.model.add(MaxPooling2D((layer[6], layer[6])))
        else:
            self.model.add(AveragePooling2D((layer[6], layer[6])))

    def __str__(self):
        str = self.hyperparameters.__str__() + "\n"
        for layer in self.iterate_layers():
            str += layer.__str__() + "\n"
        return str

    def __len__(self):
        for x in range(0, MAX_LAYERS):
            if self.genes[x][0] == 0:
                return x

    def model_summary(self):
        self.model.summary()

    def assess_fitness(self):
        self.fitness = assess_chromosome_fitness(genes=self)

