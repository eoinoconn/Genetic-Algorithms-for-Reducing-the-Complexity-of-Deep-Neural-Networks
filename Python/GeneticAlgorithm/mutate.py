from Python.GeneticAlgorithm.genes import Genes, LAYER_DEPTH, INPUT_SHAPE
import random
import logging


def mutate(genes):
    mutation_done = False

    while not mutation_done:
        rand = random.randrange(0, 3)
        logger = logging.getLogger('mutate')
        logger.info("mutating genes, rand = %f", rand)

        # remove layer
        # there should always be at least 3 layers in the genes.
        # the input convolutional, the flatten layer and the dense layer.
        if genes.__len__() > 3 and rand == 0:
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
            logger.warning("attempting unimplemented mutation")
            raise NotImplementedError


def create_parent():
    logger = logging.getLogger('mutate')
    logger.info("creating parent genes")
    genes = Genes()
    genes.add_layer(convolutional_layer(input_layer=True))
    genes.add_layer(flatten_layer())
    genes.add_layer(dense_layer())
    logger.info("parent genes created")
    return genes


# The first value of the layer array is 2 for a convolutional layer
# Other variables:
#   1   layer units
#   2   input layer
#   3   window size
#   4   activation
def convolutional_layer(input_layer=False):
    logger = logging.getLogger('mutate')
    layer = [0 for x in range(0, LAYER_DEPTH)]
    layer[0] = 2    # Sets convolutional layer
    layer[1] = 2 ** random.randrange(4, 9)     # sets layer units
    if input_layer:
        layer[2] = 1    # Sets input layer
    else:
        layer[2] = 0    # Sets hidden layer
    layer[3] = random.randrange(1, 5)   # Sets slide size
    layer[4] = set_activation()
    logger.info("added conv layer")
    return layer


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
    layer[3] = random.uniform(0.2, 0.8)
    layer[4] = set_activation()
    logger.info("added dense layer")
    return layer


# changes dropout value of a dense layer
def change_dropout_layer(genes, logger=logging.getLogger(__name__)):
    while True:
        layer_index = random.randrange(0, genes.__len__())
        layer = genes.get_layer(layer_index)
        if layer[0] == 1:   # check if dense layer
            layer[3] = random.uniform(0.2, 0.8)
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
    logger.info("adding layer type %d", layer_type)
    if layer_type == 1:     # dense layer
        new_layer = dense_layer()
        new_layer_location = random.randrange(flatten_index+1, genes.__len__())
        logger.info("added at location %d, genes length is %d", new_layer_location, genes.__len__())
    else:   # convolutional layer
        new_layer = convolutional_layer()
        new_layer_location = random.randrange(0, flatten_index+1)
        logger.info("added at location %d, genes length is %d", new_layer_location, genes.__len__())
    genes.add_layer(new_layer, new_layer_location)


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
