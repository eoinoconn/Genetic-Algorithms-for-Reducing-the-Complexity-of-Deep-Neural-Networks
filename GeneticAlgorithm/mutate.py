from GeneticAlgorithm.genes import Genes, LAYER_DEPTH, MAX_LAYERS
import random
import logging
import configparser

random.seed(1994)

EPOCHS = 2


def mutate(genes):
    mutation_done = False

    while not mutation_done:
        rand = random.randrange(0, 4)
        logger = logging.getLogger('mutate')
        logger.info("mutating genes, rand = %d", rand)
        logger.info("gene size %d", genes.__len__())

        # remove layer
        # there should always be at least 2 layers in the genes.
        # the input convolutional, the flatten layer and the dense layer.
        if rand == 0 and 2 < genes.__len__() < MAX_LAYERS:
            logger.info("removing layer")
            remove_layer(genes)
            mutation_done = True
        # add layer
        elif rand == 1:
            logger.info("adding layer")
            mutation_done = add_layer(genes)
        # change layer parameters
        elif rand == 2:
            logging.info("changing layer parameters")
            change_parameters(genes, logger)
            mutation_done = True
        # change hyperparameters
        elif rand == 3:
            logger.info("changing hyperparameters")
            mutate_hyperparameters(genes)
            mutation_done = True


def change_parameters(genes, logger):
    while True:
        layer_index = random.randrange(0, genes.__len__())
        if genes.get_layer(layer_index)[0] == 1:
            logger.info("Changing dense layer")
            change_dense_layer_parameter(genes, logger)
            break
        elif genes.get_layer(layer_index)[0] == 2:
            logger.info("Changing conv layer")
            change_conv_layer_parameter(genes, logger)
            break


def change_dense_layer_parameter(genes, logger):
    rand = random.randrange(0, 2)
    if rand == 0:     # change dense layer unit number
        logger.info("Changing dense layer unit number")
        change_dense_units(genes, logger)
    elif rand == 1:     # change dropout layer probability
        logger.info("Changing dropout")
        change_dropout_layer(genes, logger)


def change_conv_layer_parameter(genes, logger):
    rand = random.randrange(0, 3)
    if rand == 0:   # change conv layer kernel size
        logger.info("Changing convolutional kernel size")
        change_conv_kernel(genes, logger)
        return True
    elif rand == 1:     # change conv layer filter number
        logger.info("Changing convolutional filter number")
        change_conv_filter_num(genes, logger)
        return True
    elif rand == 2:     # change pooling layer
        logger.info("Changing pooling")
        return change_pooling(genes, logger)


def change_conv_kernel(genes, logger):
    min, max, interval = get_config('convolutional.layer.kernel')
    while True:
        layer_index = random.randrange(0, genes.__len__())
        layer = genes.get_layer(layer_index)
        if layer[0] == 2:   # check if conv layer
            layer[3] = random.randrange(min, max+1, interval)
            logger.info("set kernel size to %d", layer[3])
            genes.overwrite_layer(layer, layer_index)
            break


def change_dense_units(genes, logger):
    min, max, interval = get_config('convolutional.layer.kernel')
    while True:
        layer_index = random.randrange(0, genes.__len__())
        layer = genes.get_layer(layer_index)
        if layer[0] == 1:   # check if dense layer
            layer[1] = 2 ** random.randrange(min, max+1, interval)     # sets layer units
            logger.info("set unit num to %d", layer[1])
            genes.overwrite_layer(layer, layer_index)
            break


def change_conv_filter_num(genes, logger):
    min, max, interval = get_config('convolutional.layer.filter')
    while True:
        layer_index = random.randrange(0, genes.__len__())
        layer = genes.get_layer(layer_index)
        if layer[0] == 2:   # check if conv layer
            layer[1] = 2 ** random.randrange(min, max+1, interval)  # sets layer units
            logger.info("set filters to %d", layer[1])
            genes.overwrite_layer(layer, layer_index)
            break


def create_parent(input_shape):
    config = configparser.ConfigParser()
    config.read('GeneticAlgorithm/Config/training_parameters.ini')
    logger = logging.getLogger('mutate')
    logger.info("creating parent genes")

    parent = Genes(input_shape)

    if config['initial.generation'].getboolean('random_initial_generation'):
        while True:
            min, max, interval = get_config('initial.generation.conv_incep.layers')
            num_conv_layers = random.randrange(min, max+1, interval)
            min, max, interval = get_config('initial.generation.dense.layers')
            num_dense_layers = random.randrange(min, max + 1, interval)

            for i in range(0, num_conv_layers):
                parent.add_layer(convolutional_layer())

            parent.add_layer(flatten_layer())

            for i in range(0, num_dense_layers):
                parent.add_layer(dense_layer())

            if check_valid_geneset(parent, logger):
                break
            parent = Genes(input_shape)

    else:
        parent.add_layer(flatten_layer())
        parent.add_layer(dense_layer())
        logger.info("parent genes created")
        logger.info("adding hyperparameters")

    parent.set_hyperparameters(random_hyperparameters(logger))
    return parent


def random_hyperparameters(logger):
    hyperparameters = [0 for x in range(0, 4)]
    hyperparameters[0] = 'categorical_crossentropy'    # loss
    hyperparameters[1] = 'adam'                         # optimizer
    min, max, interval = get_config('chromosome.epochs')
    hyperparameters[2] = random.randrange(min, max+1, interval)     # epochs
    min, max, interval = get_config('chromosome.batchsize')
    hyperparameters[3] = random.randrange(min, max+1, interval)        # batch size
    logger.info("Set hyperparameters, loss %s, optimizer %s, epochs %d, batch size %d", hyperparameters[0],
                hyperparameters[1], hyperparameters[2], hyperparameters[3])
    return hyperparameters


def mutate_hyperparameters(genes):
    hyper_index = random.randrange(0,2)
    hyperparameters = genes.hyperparameters
    if hyper_index == 0:
        min, max, interval = get_config('chromosome.epochs')
        hyperparameters[2] = random.randrange(min, max + 1, interval)   # epochs
    else:
        min, max, interval = get_config('chromosome.batchsize')
        hyperparameters[3] = random.randrange(min, max + 1, interval)  # batch size
    genes.set_hyperparameters(hyperparameters)


# The first value of the layer array is 2 for a convolutional layer
# Other variables:
#   1   layer units
#   2   input layer
#   3   kernal size
#   4   activation
#   5   pooling type(Default 0 = None)
#   6   pool size
def convolutional_layer(input_layer=False):
    logger = logging.getLogger('mutate')
    layer = [0 for x in range(0, LAYER_DEPTH)]
    layer[0] = 2    # Sets convolutional layer
    min, max, interval = get_config('convolutional.layer.filter')
    layer[1] = 2 ** random.randrange(min, max + 1, interval)                      # sets layer units
    min, max, interval = get_config('convolutional.layer.kernel')
    layer[3] = random.randrange(min, max + 1, interval)                           # Sets slide size
    layer[4] = set_activation()
    min, max, interval = get_config('pooling.type')
    layer[5] = random.randrange(min, max + 1, interval)                           # Pooling type
    min, max, interval = get_config('pooling.filter')
    layer[6] = random.randrange(min, max + 1, interval)                           # Pooling slide size
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
    min, max, interval = get_config('pooling.type')
    layer[5] = random.randrange(min, max + 1, interval)
    min, max, interval = get_config('pooling.filter')
    layer[6] = random.randrange(min, max + 1, interval)
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
    current_dimension = genes.input_shape[0]
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
def dense_layer():
    logger = logging.getLogger('mutate')
    layer = [0 for x in range(LAYER_DEPTH)]
    layer[0] = 1
    min, max, interval = get_config('dense.layer.units')
    layer[1] = 2 ** random.randrange(min, max + 1, interval)     # sets layer units
    min, max, interval = get_config('dense.layer.dropout')
    layer[3] = (random.randrange(min, max + 1, interval)) / 10     # Set dropout probability
    layer[4] = set_activation()
    logger.info("added dense layer")
    return layer


def inception_layer():
    layer = [0 for x in range(LAYER_DEPTH)]
    layer[0] = 4
    return layer


# changes dropout value of a dense layer
def change_dropout_layer(genes, logger=logging.getLogger(__name__)):
    while True:
        layer_index = random.randrange(0, genes.__len__())
        layer = genes.get_layer(layer_index)
        if layer[0] == 1:   # check if dense layer
            min, max, interval = get_config('dense.layer.dropout')
            layer[3] = (random.randrange(min, max + 1, interval)) / 10  # Set dropout probability
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
    layer_type = random.randrange(0, 5)     # check which layer type to add
    flatten_index = genes.find_flatten()    # find the index at which the flatten layer is present

    if layer_type < 2:     # dense layer
        new_layer = dense_layer()
        new_layer_location = random.randrange(flatten_index+1, genes.__len__())
        logger.info("adding layer type dense")
    elif layer_type == 2:   # inception module
        new_layer = inception_layer()
        new_layer_location = random.randrange(0, flatten_index + 1)
        logger.info("adding layer type Inception")
    else:   # convolutional layer
        new_layer = convolutional_layer()
        new_layer_location = random.randrange(0, flatten_index+1)
        logger.info("adding layer type convolutional")
    genes.add_layer(new_layer, new_layer_location)
    logger.info("added at location %d, genes length is %d", new_layer_location, genes.__len__())
    if not check_valid_geneset(genes, logger):
        genes.remove_layer(index=new_layer_location)
        logger.info("geneset not valid, removing layer at %d", new_layer_location)
        return False
    return True


def remove_layer(genes):
    logger = logging.getLogger('mutate')
    while True:
        # randomly pick layer to remove
        layer_remove_index = random.randrange(0, genes.__len__())
        layer = genes.get_layer(layer_remove_index)
        # must not pick flatten layer which acts as border between convolutional and dense layers
        # must not pick dense or conv layer if only 1 is present
        if layer[0] == 3 or \
                (layer[0] == 1 and genes.num_dense_layers() < 2):
            continue
        genes.remove_layer(layer_remove_index)
        logger.info("removed layer type %d", layer[0])
        break


def get_config(config_name):
    config = configparser.ConfigParser()
    config.read('GeneticAlgorithm/Config/training_parameters.ini')
    config = config[config_name]
    minimum = int(config['minimum'])
    maximum = int(config['maximum'])
    interval = int(config['interval'])
    return minimum, maximum, interval