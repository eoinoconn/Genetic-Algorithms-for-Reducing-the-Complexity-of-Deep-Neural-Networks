from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Conv1D, MaxPooling2D, Dropout, Flatten, AveragePooling2D
import logging

MAX_LAYERS = 50
LAYER_DEPTH = 50
INPUT_SHAPE = (28, 28, 1)
CLASSES = 10


class Genes(object):

    def __init__(self):
        logger = logging.getLogger('fitness')
        logger.info("initialising genes")
        self.genes = [[0 for x in range(0,LAYER_DEPTH)] for y in range(0,MAX_LAYERS)]
        self.model = Sequential()

    def add_layer(self, layer, index=None):
        if index is None:
            self.genes[self.__len__()] = layer
        else:
            self.genes[index] = layer

    def overwrite_layer(self, layer, index):
        self.genes[index] = layer

    def remove_layer(self, index=None):
        if index is None:
            self.genes[self.__len__()] = [0 for x in range(0, LAYER_DEPTH)]
        else:
            self.genes[index] = [0 for x in range(0, LAYER_DEPTH)]

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

    def build_model(self):
        self.model = Sequential()
        output_layer = False
        input_layer = True
        for x in range(self.__len__()+1):
            # check if output layer, hidden layer or no layer at all
            if (self.genes[x][0] != 0) and (self.genes[x+1][0] == 0):
                output_layer = True
                self.build_layer(self.genes[x], output_layer=output_layer, input_layer=input_layer)
                input_layer = False
            elif self.genes[x][0] != 0:
                self.build_layer(self.genes[x], output_layer=output_layer, input_layer=input_layer)
                input_layer = False
            else:
                return self.model

    def build_layer(self, layer, input_layer=False, output_layer=False):
        if layer[0] == 1:   # Dense Layer
            if input_layer:           # input layer
                self.model.add(Dense(layer[1], input_shape=INPUT_SHAPE, activation='relu'))
            elif output_layer:        # output layer
                self.model.add(Dense(CLASSES, activation='softmax'))
            else:               # hidden layer
                self.model.add(Dense(layer[1], activation='relu'))
        elif layer[0] == 2:     # conv layer
            if input_layer and output_layer:    # input and output
                self.model.add(Conv2D(CLASSES, (layer[3], layer[3]), input_shape=INPUT_SHAPE, activation=layer[4]))
            elif input_layer:           # input layer
                self.model.add(Conv2D(layer[1], (layer[3], layer[3]), input_shape=INPUT_SHAPE, activation='relu'))
            elif output_layer:        # output layer
                self.model.add(Conv2D(CLASSES, (layer[3], layer[3]), activation=layer[4]))
            else:               # hidden layer
                self.model.add(Conv2D(layer[1], (layer[3], layer[3]), activation=layer[4]))
        elif layer[0] == 3:
            self.model.add(Flatten())
        else:
            raise NotImplementedError('Layers not yet implemented')

    def __str__(self):
        return self.genes.__str__()

    def __len__(self):
        for x in range(0, MAX_LAYERS):
            if self.genes[x][0] == 0:
                return x

    def model_summary(self):
        self.model.summary()
