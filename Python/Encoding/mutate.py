from Python.Encoding.chromosome import Chromosome, LAYER_DEPTH, INPUT_SHAPE
import random


def mutate(chromosome, input_layer=False):
    rand = random.randrange(0, 2)
    # remove layer
    if len(chromosome) > 2 and rand == 0:
        # Remove Gene
        chromosome.remove_layer()

    # add layer
    elif rand == 1:
        rand = random.randrange(0, 2)
        if True:
            chromosome.add_layer(dense_layer())
        elif rand == 1:
            chromosome.add_layer(convolutional_layer())
    # change hyperparameter
    elif rand == 2:
        raise NotImplementedError

def create_parent():
    chromosome = Chromosome()
    chromosome.add_layer(convolutional_layer(input_layer=True))
    chromosome.add_layer(flatten_layer())
    chromosome.add_layer(dense_layer())
    return chromosome


def convolutional_layer(input_layer=False):
    layer = [0 for x in range(0, LAYER_DEPTH)]
    layer[0] = 2    # Sets convolutional layer
    layer[1] = random.randrange(1, 16) * 16     # sets layer units
    if input_layer:
        layer[2] = 1    # Sets input layer
    else:
        layer[2] = 0    # Sets hidden layer
    layer[3] = random.randrange(1, 5)   # Sets slide size
    layer[4] = set_activation()
    return layer


def flatten_layer():
    layer = [0 for x in range(0, LAYER_DEPTH)]
    layer[0] = 3
    return layer


def dense_layer(input_layer=False):
    layer = [0 for x in range(LAYER_DEPTH)]
    layer[0] = 1
    layer[1] = random.randrange(1, 16) * 16  # sets layer units
    if input_layer:
        layer[2] = 1    # Sets input layer
    else:
        layer[2] = 0    # Sets hidden layer
    layer[4] = set_activation()


def set_activation():
    ran_active = random.randrange(0, 3)
    if ran_active == 0:
        return 'relu'
    elif ran_active == 1:
        return 'sigmoid'
    elif ran_active == 2:
        return 'linear'
    else:
        return 'elu'

