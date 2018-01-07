from Python.Encoding.chromosome import Chromosome, LAYER_DEPTH, INPUT_SHAPE
import random
import logging


def mutate(chromosome):
    logging.info("mutating chromosome")
    rand = random.randrange(0, 2)
    # remove layer
    # there should always be at least 3 layers in the chromosome.
    # the input convolutional, the flatten layer and the dense layer.

    # remove layer
    if chromosome.__len__() > 3 and rand == 0:
        logging.info("removing layer")
        remove_layer(chromosome)
    # add layer
    elif rand == 1:
        logging.info("adding layer")
        add_layer(chromosome)
    # change layer
    elif rand == 2:
        logging.warning("attempting unimplemented mutation")
        raise NotImplementedError
    # change hyperparameter
    elif rand == 3:
        logging.warning("attempting unimplemented mutation")
        raise NotImplementedError


def create_parent():
    logging.info("creating parent chromosome")
    chromosome = Chromosome()
    chromosome.add_layer(convolutional_layer(input_layer=True))
    chromosome.add_layer(flatten_layer())
    chromosome.add_layer(dense_layer())
    logging.info("parent chromosome created")
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
    return layer


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


def add_layer(chromosome):
    change_layer = False
    while not change_layer:
        rand = random.randrange(0, chromosome.__len__())
        layer = chromosome.get_layer(rand)
        if layer[0] == 1:
            new_layer = dense_layer()
            change_layer = True
        elif layer[0] == 2:
            new_layer = convolutional_layer()
            change_layer = True
        else:
            continue

    for i in range(rand+1, chromosome.__len__()+1):
        temp_layer = chromosome.get_layer(i)
        chromosome.add_layer(new_layer, i)
        new_layer = temp_layer


def remove_layer(chromosome):
    while True:
        rand = random.randrange(0, chromosome.__len__())
        layer = chromosome.get_layer(rand)
        if layer[0] == 3:
            continue
        chromosome.remove_layer(rand)
        for i in range(rand+1, chromosome.__len__()):
            temp_layer = chromosome.get_layer(i+1)
            chromosome.add_layer(temp_layer, i)
        break
