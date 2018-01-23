from keras.utils import to_categorical
from Python.GeneticAlgorithm.mutate import create_parent, mutate
from keras.datasets import mnist
import operator
import logging
import numpy as np

POOL_SIZE = 6
IMAGE_SIZE = 28
NUM_LABELS = 10


def get_best(max_generations, fn_unpack_training_data):

    logger = logging.getLogger('geneticEngine')
    logger.info('Starting genetic engine...')

    training_data = fn_unpack_training_data()

    generation = 1
    population = create_population(logger)
    logger.info("Generation number: %d", generation)
    while generation < max_generations:
        assess_population_fitness(population, training_data, logger)
        mutate_population(population, logger)
        generation += 1


def create_population(logger):
    pool = []
    for x in range(POOL_SIZE):
        pool.append(create_parent())
        logger.info("Added chromosome number %d to population", x+1)
    return pool


def assess_population_fitness(population, training_data, logger):
    i = 0
    for chromosome in population:
        logger.info("getting fitness of chromosome %d", i+1)
        chromosome.assess_fitness(training_data)
        i += 1
    population.sort(key=operator.attrgetter('fitness'))


def mutate_population(population, logger):
    i = 0
    for chromosome in population:
        logger.info("mutating chromosome %d", i+1)
        mutate(chromosome)
        i += 1





