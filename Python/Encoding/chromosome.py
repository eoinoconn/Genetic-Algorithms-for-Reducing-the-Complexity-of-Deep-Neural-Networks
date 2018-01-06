from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Conv1D, MaxPooling2D, Dropout, Flatten, AveragePooling2D

MAX_LAYERS = 50
LAYER_DEPTH = 50
INPUT_SHAPE = (28, 28, 1)
CLASSES = 10


class Chromosome(object):

    def __init__(self):
        self.chromosome = [[0 for x in range(LAYER_DEPTH)] for y in range(MAX_LAYERS)]
        self.model = Sequential()

    def add_layer(self, layer):
        self.chromosome[self.__len__()] = layer

    def remove_layer(self, index=None):
        if index is None:
            self.chromosome[self.__len__()].remove
        else:
            raise NotImplementedError("remove layer at index not implemented")

    def build_model(self):
        self.model.layers.clear()
        output_layer = False
        input_layer = True
        for x in range(self.chromosome.__len__()):
            # check if output layer, hidden layer or no layer at all
            if (self.chromosome[x][0] != 0) and (self.chromosome[x+1][0] == 0):
                output_layer = True
                self.build_layer(self.chromosome[x], output_layer=output_layer, input_layer=input_layer)
                input_layer = False
            elif self.chromosome[x][0] != 0:
                self.build_layer(self.chromosome[x], output_layer=output_layer, input_layer=input_layer)
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
        elif layer[0] == 2:
            if input_layer:           # input layer
                self.model.add(Conv2D(layer[1], (layer[3], layer[3]), input_shape=INPUT_SHAPE, activation='relu'))
            elif output_layer:        # output layer
                self.model.add(Conv2D(CLASSES, (layer[3], layer[3]), activation=layer[4]))
            else:               # hidden layer
                self.model.add(Conv2D(layer[1], (layer[3], layer[3]), activation=layer[4]))
        else:
            raise NotImplementedError('Layers other than dense have not been implemented')

    def __str__(self):
        return self.chromosome.__str__()

    def __len__(self):
        for x in range(0, MAX_LAYERS):
            if self.chromosome[x][0] == 0:
                return x

    def model_summary(self):
        self.model.summary()
