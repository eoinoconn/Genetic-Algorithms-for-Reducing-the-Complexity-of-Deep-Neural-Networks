from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, AveragePooling2D, \
    concatenate, Input, BatchNormalization, Activation
from keras.models import Model


class ChromosomeModel(object):

    def __init__(self, genes, classes, input_shape, length):
        self.genes = genes
        self.classes = classes
        self.input_shape = input_shape
        self.length = length

    def build_model(self, logger):
        input_layer = Input(shape=self.input_shape)
        model = input_layer
        for x in range(self.length + 1):
            # check if output layer, hidden layer or no layer at all
            if (self.genes[x][0] != 0) and (self.genes[x + 1][0] == 0):  # Check if output layer
                model = self.build_layer(model, self.genes[x], logger, output_layer=True)
            elif self.genes[x][0] != 0:  # else check if not empty layer
                model = self.build_layer(model, self.genes[x], logger)
            else:
                return Model(inputs=input_layer, outputs=model)

    def build_layer(self, model, layer, logger, output_layer=False):
        if layer[0] == 1:                   # dense Layer
            if output_layer:            # output layer
                return Dense(self.classes, activation='softmax')(model)
            else:                       # hidden layer
                if layer[-1] is not 0:
                    model = Dense(layer[1])(model).set_weights(layer[-1])
                else:
                    model = Dense(layer[1])(model)
                if layer[4] is not None:
                    model = self.activation(layer, model)
                if layer[3] > 0:    # Dropout layer
                    model = Dropout(layer[3])(model)
                logger.info("output dimensions %d", model.shape[1])
                return model

        elif layer[0] == 2:                 # convolutional layer
            model = Conv2D(layer[1], layer[3], strides=layer[2], padding=layer[7])(model)
            logger.info("output dimensions (%d, %d)", model.shape[1], model.shape[2])
            if layer[10] == 1:       # Batch normalisation layer
                if layer[-2] is not 0:
                    model = self.batch_normalisation(model).set_weights(layer[-2])
                else:
                    model = self.batch_normalisation(model)
            if layer[4] is not None:
                model = self.activation(layer, model)
            if layer[9] > 0:    # Dropout layer
                model = Dropout(layer[9])(model)
            if layer[5] > 0:        # Pooling layer
                model = self.pooling_layer(model, layer)
                logger.info("output dimensions (%d, %d)", model.shape[1], model.shape[2])
            return model

        elif layer[0] == 3:                 # Flatten layer
            return Flatten()(model)

        elif layer[0] == 4:                 # Inception layer
            return self.inception_module(model, layer)

        else:
            raise NotImplementedError('Layers not yet implemented')

    @staticmethod
    def activation(layer, model):
        return Activation(layer[4])(model)

    @staticmethod
    def batch_normalisation(model):
        return BatchNormalization()(model)

    @staticmethod
    def pooling_layer(input_layer, layer):
        if layer[5] == 1:  # max pooling
            return MaxPooling2D((layer[6], layer[6]), strides=layer[8])(input_layer)
        else:
            return AveragePooling2D((layer[6], layer[6]), strides=layer[8])(input_layer)

    @staticmethod
    def inception_module(input_layer, layer):
        if layer[-1] is not 0:
            inception_weights_and_biases = layer[-1]
        tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_layer)
        tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)
        tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_layer)
        tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)
        tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_layer)
        tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)
        return concatenate([tower_1, tower_2, tower_3], axis=3)
