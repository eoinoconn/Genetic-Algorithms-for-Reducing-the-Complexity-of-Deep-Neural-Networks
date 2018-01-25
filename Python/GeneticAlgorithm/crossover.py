import copy
import logging
from random import randrange

from Python.GeneticAlgorithm.genes import Genes
from Python.GeneticAlgorithm.mutate import flatten_layer


def crossover(parent_1, parent_2):

    logger = logging.getLogger('crossover')
    logger.info("breeding parents")
    logger.info("Parnet 1 id: %d\tParent 2 id:%d", parent_1.id, parent_2.id)

    child_conv_layers = parent_1.num_conv_layers()
    child_dense_layers = parent_1.num_dense_layers()

    copy_parent_1 = copy.deepcopy(parent_1)
    copy_parent_2 = copy.deepcopy(parent_2)

    child = Genes()
    logger.info("Child gene has id %d", child.id)

    for i in range(0, child_conv_layers):
        rand = randrange(0, 2)
        if rand == 0:
            layer = copy_parent_2.get_layer(0)
            copy_parent_1.remove_layer(0)
            copy_parent_2.remove_layer(0)
        else:
            layer = copy_parent_1.get_layer(0)
            copy_parent_1.remove_layer(0)
            copy_parent_2.remove_layer(0)

        child.overwrite_layer(layer, i)

    child.add_layer(flatten_layer())
    copy_parent_1.remove_layer(0)
    copy_parent_2.remove_layer(0)

    for i in range(0, child_dense_layers):
        rand = randrange(0, 2)
        if rand == 0:
            layer = copy_parent_2.get_layer(0)
            copy_parent_1.remove_layer(0)
            copy_parent_2.remove_layer(0)
        else:
            layer = copy_parent_1.get_layer(0)
            copy_parent_1.remove_layer(0)
            copy_parent_2.remove_layer(0)

        child.overwrite_layer(layer, i+child_conv_layers+1)

    child.set_hyperparameters(parent_1.hyperparameters)
    child.log_geneset()

    return child
