from keras.utils import to_categorical

from GeneticAlgorithm.crossover import crossover
from GeneticAlgorithm.mutate import create_parent, mutate
from GeneticAlgorithm.utils import *
from keras.datasets import cifar10, mnist
import operator
import logging
import random
import configparser
import copy
from keras import backend as K
import io
import numpy as np


def get_best(max_generations, input_shape, training_data):
    """ Main Genetic algorithm loop, performing high level operations and calling
    functions
    :rtype: chromosome""" 

    logger = logging.getLogger('geneticEngine')
    logger.info('Starting genetic engine...')

    config = setup_global_variables()

    setup_csvlogger()

    trained_chromosomes = {}

    generation = 1
    population = create_population(input_shape, config, logger)
    best_chromosome = population[0]

    while generation < max_generations:
        logger.info("Generation number: %d", generation)
        logger.info("pool size: %d", population.__len__())

        # Assign population fitness
        best_child = assess_population_fitness(population, training_data, trained_chromosomes, logger)

        # if new best chromosome found, save it
        if best_child > best_chromosome:
            logger.info("New best child, id: %d", best_child.id)
            best_chromosome = copy.deepcopy(best_child)
            best_chromosome.assess_fitness_with_test(training_data, log_csv=True)
            best_chromosome.log_best()

        intermittent_logging(best_child, generation)

        # select best chromosomes
        population.extend(spawn_children(population, input_shape, logger))

        # remove poorest performers
        population = population[(population.__len__() - POOL_SIZE):]

        # mutate pool
        mutate_population(population, logger)

        # iterate the age of every chromosome in the population by 1
        age_population(population)

        logger.info("End of generation %d \n\n", generation)
        generation += 1
    return best_chromosome


def setup_global_variables():
    """initialises global variables from configuration file
    """
    config = configparser.ConfigParser()
    config.read('GeneticAlgorithm/Config/training_parameters.ini')

    global POOL_SIZE, MAX_CROSSOVERS
    POOL_SIZE = int(config['genetic.engine']['pool_size'])
    MAX_CROSSOVERS = int(config['genetic.engine']['max_crossover'])
    return config


def create_population(input_shape, config, logger):
    """ Create the population pool of chromosomes

    :param config:
    :param input_shape:
    :param logger:
    :return:
    """
    pool = []
    if config['search.known.architecture'].getboolean('enable'):
        known_architecture = load_known_architecture(config['search.known.architecture']['file_name'], input_shape)
        for x in range(POOL_SIZE):
            pool.append(copy.deepcopy(known_architecture))
    else:
        for x in range(POOL_SIZE):
            pool.append(create_parent(input_shape))
            logger.info("Added chromosome number %d to population", pool[x].id)
    return pool


def assess_population_fitness(population, training_data, assessed_list, logger):
    """ Assesses the fitness of each chromosome

    :param population:
    :param training_data:
    :param assessed_list:
    :param logger:
    :return chromosome:
    """
    for chromosome in population:
        chromosome.log_geneset()
        mash_value = chromosome.mash()
        if mash_value in assessed_list:
            # chromosome already trained
            logger.info("chromosome %d already trained as chromosome %d, age %d", chromosome.id,
                        assessed_list[mash_value][3],
                        assessed_list[mash_value][4])
            chromosome.assume_values(assessed_list[mash_value])
        else:
            # chromosome not trained before
            logger.info("getting fitness of chromosome %d", chromosome.id)
            chromosome.assess_fitness(training_data)
            add_assessed_to_dict(chromosome, assessed_list)
    population.sort(key=operator.attrgetter('fitness'))
    return population[-1]


def add_assessed_to_dict(chromosome, assessed_list):
    assessed_list[chromosome.mash()] = [chromosome.fitness, chromosome.accuracy,
                                        chromosome.parameters, chromosome.id, chromosome.age]


def mutate_population(population, logger):
    for chromosome in population[:(POOL_SIZE-(MAX_CROSSOVERS*4))]:
        logger.info("mutating chromosome %d", chromosome.id)
        mutate(chromosome)


def spawn_children(population, input_shape, logger):
    """ Perform crossover on parent couples

    :param population:
    :param input_shape:
    :param logger:
    :return:
    """
    child_chromosomes = []
    for i in range(0, (MAX_CROSSOVERS*2), 2):
        if population.__len__() < 2:
            break
        parent_1 = population[-i]
        parent_2 = population[-(i+1)]
        logger.info("Spawning children from chromosomes %d and %d", parent_1.id, parent_2.id)
        child_chromosomes.append(crossover(parent_1, parent_2, input_shape))
        child_chromosomes.append(crossover(parent_2, parent_1, input_shape))
    return child_chromosomes


def age_population(population):
    for chromosome in population:
        chromosome.increment_age()
