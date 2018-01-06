from Python.Encoding.chromosome import Chromosome, LAYER_DEPTH, INPUT_SHAPE
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


def convolutional_layer(input = False):
    layer = [0 for x in range(LAYER_DEPTH)]
    layer[0] = 2    # Sets convolutional layer
    layer[1] = random.randrange(1, 16) * 16 # sets layer units
    if input:
        layer[2] = 1    # Sets input layer
    else:
        layer[2] = 0    # Sets hidden layer
    layer[3] = random.randrange(1, 5)   # Sets slide size
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

