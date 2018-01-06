from Python.Encoding.chromosome import Chromosome, LAYER_DEPTH
import random


def mutate(chromosome, input_layer=False):
    rand = random.randrange(0, 3)
    if len(chromosome) > 2 and rand % 2 == 0:

        # Remove Gene
        chromosome.remove_layer()
    else:

        # Add Gene
        ran = random.randrange(1, 16) * 16
        ran_active = random.randrange(0, 5)
        if ran_active == 0:
            activation = 'relu'
        elif ran_active == 1:
            activation = 'sigmoid'
        elif ran_active == 2:
            activation = 'softmax'
        elif ran_active == 3:
            activation = 'linear'
        elif ran_active == 4:
            activation = 'elu'
        else:
            activation = 'softsign'

        if not input_layer:
            gene = DenseGene(ran, activation=activation)
        else:
            gene = DenseGene(ran, activation=activation, input_layer=True)

        chromosome.add_gene(gene)
        return chromosome

def create_parent():
    chromosome = Chromosome()
    chromosome.add_layer(convolutional_layer())
    return chromosome

def convolutional_layer():
    layer = [0 for x in range(LAYER_DEPTH)]
    layer[0] = 1
    