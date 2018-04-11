from keras.utils import to_categorical

from GeneticAlgorithm.crossover import crossover
from GeneticAlgorithm.chromosome import Chromosome
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

    # trained_chromosomes = {}

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
        population = cleanse_population(population, logger)

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
        raise NotImplementedError
    else:
        for x in range(POOL_SIZE):
            pool.append(Chromosome(input_shape))
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
        logger.info("getting fitness of chromosome %d", chromosome.id)
        chromosome.assess(training_data)
        logger.info("fitness: %f, Accuracy: %f, Parameters: %d",
                    chromosome.fitness,
                    chromosome.accuracy,
                    chromosome.parameters)
    population.sort(key=operator.attrgetter('fitness'))
    return population[-1]


def mutate_population(population, logger):
    for chromosome in population[:(POOL_SIZE-(MAX_CROSSOVERS*4))]:
        logger.info("mutating chromosome %d", chromosome.id)
        chromosome.mutate()


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
        logger.info("child spawn has id %d", child_chromosomes[-1].id)
        child_chromosomes.append(crossover(parent_2, parent_1, input_shape))
        logger.info("child spawn has id %d", child_chromosomes[-1].id)
    return child_chromosomes


def cleanse_population(population, logger):
    for chromosome in population[:(population.__len__() - POOL_SIZE)]:
        logger.info("removing chromosome %d age %d", chromosome.id, chromosome.age)
    population = population[(population.__len__() - POOL_SIZE):]
    logger.info("Chromosomes still in population")
    for chromosome in population:
        logger.info("ID %d,  Age %d", chromosome.id, chromosome.age)
    return population


def age_population(population):
    for chromosome in population:
        chromosome.increment_age()
