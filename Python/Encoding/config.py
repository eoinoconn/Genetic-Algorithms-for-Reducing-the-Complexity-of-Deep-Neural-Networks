class Config(object):

    def __init__(self, num_inputs, input_shape, learning_rate, num_labels):
        self.num_inputs = num_inputs
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.num_labels = num_labels

    def get_num_inputs(self):
        return self.num_inputs

    def get_input_shape(self):
        return self.input_shape

    def get_learning_rate(self):
        return self.learning_rate

    def get_num_labels(self):
        return self.num_labels

