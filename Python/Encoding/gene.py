from keras.layers import Dense, Activation, Conv2D, Conv1D, MaxPooling1D, Dropout, Flatten


class BaseGene(object):

    def compile(self, model):
        raise NotImplementedError("Please Implement this method")

    def build_details(self):
        raise NotImplementedError("Please Implement this method")

    def set_activation(self, activation):
        self.activation = activation

class Gene(object):

    def __init__(self, gene_config):
        self.gene_config = gene_config

        self.layer_type = self.gene_config.layer_type

        if self.gene_config.input_layer:
            self.input_layer = True
            self.input_shape = self.gene_config.input_shape

    def compile(self, model):
        details, kwag = self.build_details()
        if self.layer_type == "Dense":
            model.add(Dense(*details, **kwag))
        elif self.layer_type == "Convolutional":
            model.add(Conv2D(*details, **kwag))
        elif self.layer_type == 'Flatten':
            model.add(Flatten())
        return model

    def build_details(self):
        details = []
        kwag = {}
        if self.layer_type == "Convolutional":
            details = [self.gene_config.layer_units, self.gene_config.kernel_size]
        elif self.layer_type == "Dense":
            details = [self.gene_config.layer_units]
        if self.gene_config.input_layer:
            kwag['input_shape'] = self.input_shape
        kwag['activation'] = self.gene_config.activation

        return details, kwag


class DenseGene(BaseGene):

    def __init__(self, layer_units, input_layer=False, input_shape=(1, 1, 1),
                 activation="relu"):

        self.layer_type = "Dense"
        self.layer_units = layer_units
        self.activation = activation
        self.input_layer = input_layer

        if input_layer:
            self.input_shape = input_shape

    def compile(self, model):
        details, kwag = self.build_details()
        model.add(Dense(*details, **kwag))
        return model

    def build_details(self):
        kwag = {}
        details = [self.layer_units]
        if self.input_layer:
            kwag['input_shape'] = self.input_shape
        kwag['activation'] = self.activation

        return details, kwag


class ConvolutionalGene(BaseGene):
    def __init__(self, layer_units, kernel_size=(2, 2), input_layer=False, input_shape=(28, 28, 1),
                 activation="relu"):

        self.layer_type = "Convolutional"
        self.layer_units = layer_units
        self.kernel_size = kernel_size
        self.activation = activation
        self.input_layer = input_layer

        if input_layer:
            self.input_shape = input_shape

    def compile(self, model):
        details, kwag = self.build_details()
        model.add(Conv2D(*details, **kwag))
        return model

    def build_details(self):
        kwag = {}
        details = [self.layer_units] + [self.kernel_size]
        if self.input_layer:
            kwag['input_shape'] = self.input_shape
        kwag['activation'] = self.activation

        return details, kwag

class GeneConfig(object):

    def __init__(self, layer_type, layer_units=64, input_layer=False, input_shape=(1, 1, 1),
                 kernel_size=(2, 2), activation="relu", padding="same"):
        self.layer_type = layer_type
        self.layer_units = layer_units
        self.input_layer = input_layer
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding


