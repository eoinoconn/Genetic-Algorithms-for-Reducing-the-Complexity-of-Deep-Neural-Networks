from keras.layers import Dense, Activation, Conv2D, Conv1D, MaxPooling1D, Dropout, Flatten

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


class GeneConfig(object):

    def __init__(self, layer_type, layer_units, input_layer, input_shape, kernel_size, activation, padding):
        self.layer_type = layer_type
        self.layer_units = layer_units
        self.input_layer = input_layer
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding


