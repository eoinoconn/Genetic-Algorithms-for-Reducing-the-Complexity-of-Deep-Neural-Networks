from Python.GeneticAlgorithm.genes import Genes, LAYER_DEPTH, INPUT_SHAPE, MAX_LAYERS
import random
import logging


def mutate(genes):
    mutation_done = False

    while not mutation_done:
        rand = random.randrange(0, 5)
        logger = logging.getLogger('mutate')
        logger.info("mutating genes, rand = %f", rand)

        # remove layer
        # there should always be at least 3 layers in the genes.
        # the input convolutional, the flatten layer and the dense layer.
        if  rand == 0 and genes.__len__() > 3 and genes.__len__() < MAX_LAYERS:
            logger.info("removing layer")
            remove_layer(genes)
            mutation_done = True
        # add layer
        elif rand == 1:
            logger.info("adding layer")
            add_layer(genes)
            mutation_done = True
        # change dropout
        elif rand == 2:
            logger.info("changing dropout")
            change_dropout_layer(genes, logger)
            mutation_done = True
        # change pooling layer
        elif rand == 3:
            logger.info("changing pooling")
            mutation_done = change_pooling(genes, logger)
        elif rand == 4:
            logger.info("changing hyperparameters")
            mutate_hyperparameters(genes)
            mutation_done = True

        # iterate genes id
        genes.iterate_id()


def create_parent():
    logger = logging.getLogger('mutate')
    logger.info("creating parent genes")
    while True:
        genes = Genes()
        parent_size = random.randrange(4, 7)
        flatten_layer_index = random.randrange(1, parent_size - 1)
        for i in range(0, parent_size):
            if i < flatten_layer_index:
                genes.add_layer(convolutional_layer())
            elif i == flatten_layer_index:
                genes.add_layer(flatten_layer())
            else:
                genes.add_layer(dense_layer())
        if check_valid_geneset(genes, logger):
            break
    logger.info("parent genes created")
    logger.info("adding hyperparameters")
    genes.set_hyperparameters(random_hyperparameters(logger))
    return genes


def random_hyperparameters(logger):
    hyperparameters = [0 for x in range(0, 25)]
    hyperparameters[0] = 'categorical_crossentropy'    # loss
    hyperparameters[1] = 'adam'                         # optimizer
    hyperparameters[2] = 15   # epochs
    hyperparameters[3] = random.randrange(50, 200, 25)  # batch size
    logger.info("Set hyperparameters, loss %s, optimizer %s, epochs %d, batch size %d", hyperparameters[0],
                hyperparameters[1], hyperparameters[2], hyperparameters[3])
    return hyperparameters


def mutate_hyperparameters(genes):
    hyper_index = random.randrange(0,2)
    hyperparameters = genes.hyperparameters
    if hyper_index == 0:
        hyperparameters[2] = 15   # epochs
    else:
        hyperparameters[3] = random.randrange(50, 200, 25)  # batch size
    genes.set_hyperparameters(hyperparameters)


# The first value of the layer array is 2 for a convolutional layer
# Other variables:
#   1   layer units
#   2   input layer
#   3   window size
#   4   activation
#   5   pooling type(Default 0 = None)
#   6   pool size
def convolutional_layer(input_layer=False):
    logger = logging.getLogger('mutate')
    layer = [0 for x in range(0, LAYER_DEPTH)]
    layer[0] = 2    # Sets convolutional layer
    layer[1] = 2 ** random.randrange(4, 9)     # sets layer units
    if input_layer:
        layer[2] = 1    # Sets input layer
    else:
        layer[2] = 0    # Sets hidden layer
    layer[3] = random.randrange(1, 6)   # Sets slide size
    layer[4] = set_activation()
    layer[5] = random.randrange(1, 3)
    layer[6] = random.randrange(2, 5)
    logger.info("added conv layer")
    return layer


# Adds pooling to genes
#   0   no pooling (Default)
#   1   max pooling
#   2   avg pooling
def change_pooling(genes, logger):
    # first we need to pick a convolutional layer
    flatten_index = genes.find_flatten()
    conv_layer_index = random.randrange(0, flatten_index)
    layer = genes.get_layer(conv_layer_index)
    temporary_values = [layer[5], layer[6]]
    layer[5] = random.randrange(1, 3)
    layer[6] = random.randrange(2, 5)
    if check_valid_geneset(genes, logger):
        logger.info("Setting pooling in layer %d to type %d with pool size %d", conv_layer_index, layer[5], layer[6])
        return True
    else:
        layer[5] = temporary_values[0]
        layer[6] = temporary_values[1]
        logger.info("no pooling changes have occurred")
        return False


# This is a function to check if the mutated geneset has
# valid dimensions after pooling is altered
# it does this by calculating the smallest dimension of the 
# geneset at the last convolutional layer
def check_valid_geneset(genes, logger=logging.getLogger(__name__)):

    current_dimension = INPUT_SHAPE[0]
    logger.info("checking for valid geneset; conv dimensions %d", current_dimension)
    for layer in genes.iterate_layers():
        if layer[0] == 2:
            current_dimension -= (layer[3] - 1)
            if layer[5] > 0:
                current_dimension = int(current_dimension/layer[6])
        elif layer[0] == 3:
            break
    if current_dimension < 1:   # invalid geneset
        logger.info("Invalid geneset, dimensions less than 0, Dimension: %d", current_dimension)
        return False
    else:
        logger.info("valid geneset found, min dimension: %d", current_dimension)
        return True         # valid geneset


def flatten_layer():
    layer = [0 for x in range(0, LAYER_DEPTH)]
    layer[0] = 3
    return layer


# The first value of the layer array is 1 for a dense layer
# Other variables:
#   1   layer units
#   2   input layer
#   3   dropout
#   4   activation
def dense_layer(input_layer=False):
    logger = logging.getLogger('mutate')
    layer = [0 for x in range(LAYER_DEPTH)]
    layer[0] = 1
    layer[1] = 2 ** random.randrange(4, 9)     # sets layer units
    if input_layer:
        layer[2] = 1    # Sets input layer
    else:
        layer[2] = 0    # Sets hidden layer
    layer[3] = random.uniform(0.2, 0.5)     # Set dropout probability
    layer[4] = set_activation()
    logger.info("added dense layer")
    return layer


# changes dropout value of a dense layer
def change_dropout_layer(genes, logger=logging.getLogger(__name__)):
    while True:
        layer_index = random.randrange(0, genes.__len__())
        layer = genes.get_layer(layer_index)
        if layer[0] == 1:   # check if dense layer
            layer[3] = random.uniform(0.2, 0.5)
            logger.info("set droupout to %f", layer[3])
            genes.overwrite_layer(layer, layer_index)
            break


def set_activation():
    ran_active = random.randrange(0, 3)
    if True:
        return 'relu'
    elif ran_active == 1:
        return 'sigmoid'
    elif ran_active == 2:
        return 'linear'
    else:
        return 'elu'


def add_layer(genes):
    logger = logging.getLogger('mutate')
    layer_type = random.randrange(1, 3)     # check which layer type to add
    flatten_index = genes.find_flatten()    # find the index at which the flatten layer is present

    if layer_type == 1:     # dense layer
        new_layer = dense_layer()
        new_layer_location = random.randrange(flatten_index+1, genes.__len__())
        logger.info("adding layer type dense")
    else:   # convolutional layer
        new_layer = convolutional_layer()
        new_layer_location = random.randrange(0, flatten_index+1)
        logger.info("adding layer type convolutional")
    genes.add_layer(new_layer, new_layer_location)
    logger.info("added at location %d, genes length is %d", new_layer_location, genes.__len__())
    if not check_valid_geneset(genes, logger):
        genes.remove_layer(index=new_layer_location)
        logger.info("geneset not valid, removing layer at %d", new_layer_location)


def remove_layer(genes):
    logger = logging.getLogger('mutate')
    while True:
        # randomly pick layer to remove
        layer_remove_index = random.randrange(0, genes.__len__())
        layer = genes.get_layer(layer_remove_index)
        # must not pick flatten layer which acts as border between convolutional and dense layers
        # must not pick dense or conv layer if only 1 is present
        if layer[0] == 3 or \
                (layer[0] == 2 and genes.num_conv() < 2) or \
                (layer[0] == 1 and genes.num_dense() < 2):
            continue
        genes.remove_layer(layer_remove_index)
        logger.info("removed layer type %d", layer[0])
        break

