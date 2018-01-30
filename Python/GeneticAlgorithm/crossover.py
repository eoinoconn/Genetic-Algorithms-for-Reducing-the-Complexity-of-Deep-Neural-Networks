import copy
import logging
from random import randrange

from Python.GeneticAlgorithm.genes import Genes
from Python.GeneticAlgorithm.mutate import flatten_layer


def crossover(dom_parent, parent_2):

    logger = logging.getLogger('crossover')
    logger.info("breeding parents")
    logger.info("Dominant parent 1 id: %d\tSecond parent 2 id:%d", dom_parent.id, parent_2.id)

    child_conv_layers = dom_parent.num_conv_layers()
    child_dense_layers = dom_parent.num_dense_layers()

    copy_dom_parent_1 = copy.deepcopy(dom_parent)
    copy_parent_2 = copy.deepcopy(parent_2)

    child = Genes()
    logger.info("Child gene has id %d", child.id)
    logger.info("conv layers %d, dense layers: %d", child_conv_layers, child_dense_layers)

    copy_to_child(child, copy_dom_parent_1, copy_parent_2, 2, child_conv_layers, 0, logger)

    child.add_layer(flatten_layer())
    delete_to_flatten(copy_dom_parent_1)
    delete_to_flatten(copy_parent_2)

    copy_to_child(child, copy_dom_parent_1, copy_parent_2, 1, child_dense_layers, child_conv_layers+1, logger)

    child.set_hyperparameters(dom_parent.hyperparameters)
    child.log_geneset(log_file='crossover')

    return child


def delete_to_flatten(chromosome):
    flatten_index = chromosome.find_flatten()
    if flatten_index == -1:
        chromosome.log_geneset('crossover')
        raise Exception("flatten not present in chromosome")
    while flatten_index != 0:
        chromosome.remove_layer(0)
        flatten_index = chromosome.find_flatten()
    chromosome.remove_layer(0)


def copy_to_child(child, dom_parent, parent_2, layer_type, num_layers, child_index_offset, logger):
    for i in range(0, num_layers):
        logger.info("Layer type: %d\tlayer num: %d",layer_type, i+1)
        parameters_added = False
        layer = []
        while not parameters_added:
            rand = randrange(1, 3)
            logger.info("Choosing parent %d", rand)
            if rand == 2 and parent_2.get_layer(0)[0] == layer_type:
                layer = parent_2.get_layer(0)
                parent_2.remove_layer(0)
                parameters_added = True
            elif rand == 1 and dom_parent.get_layer(0)[0] == layer_type:
                layer = dom_parent.get_layer(0)
                dom_parent.remove_layer(0)
                parameters_added = True
        child.overwrite_layer(layer, i + child_index_offset)
