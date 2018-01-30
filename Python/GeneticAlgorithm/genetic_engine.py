from keras.utils import to_categorical

from Python.GeneticAlgorithm.crossover import crossover
from Python.GeneticAlgorithm.mutate import create_parent, mutate
from keras.datasets import mnist
import operator
import logging
from random import shuffle
import numpy as np

POOL_SIZE = 10
IMAGE_SIZE = 28
NUM_LABELS = 10
MAX_CROSSOVERS = 2


def get_best(max_generations, fn_unpack_training_data):

    logger = logging.getLogger('geneticEngine')
    logger.info('Starting genetic engine...')

    training_data = fn_unpack_training_data()

    generation = 1
    population = create_population(logger)
    best_chromosome = population[0]

    while generation < max_generations:
        logger.info("Generation number: %d", generation)
        logger.info("pool size: %d", population.__len__())

        # iterate the age of every chromosome in the population by 1
        age_population(population)

        # Assign population fitness
        best_child = assess_population_fitness(population, training_data, logger)

        # if new best chromosome found, save it
        if best_child > best_chromosome:
            best_chromosome = best_child
            best_chromosome.log_best()

        # select best chromosomes
        population.extend(spawn_children(population, logger))

        # mutate pool
        mutate_population(population, logger)

        logger.info("End of generation %d \n\n", generation)
        generation += 1
    return best_chromosome


def create_population(logger):
    pool = []
    for x in range(POOL_SIZE):
        pool.append(create_parent())
        logger.info("Added chromosome number %d to population", x+1)
    return pool


def assess_population_fitness(population, training_data, logger):
    for chromosome in population:
        logger.info("getting fitness of chromosome %d", chromosome.id)
        chromosome.assess_fitness(training_data)
    population.sort(key=operator.attrgetter('fitness'))
    return population[-1]


def mutate_population(population, logger):
    mutations_completed = 0
    shuffle(population)
    for chromosome in population:
        logger.info("mutating chromosome %d", chromosome.id)
        mutate(chromosome)
        mutations_completed += 1


def spawn_children(population, logger):
    child_chromosomes = []
    spawned_children = 0
    while population.__len__() > 1 and spawned_children < MAX_CROSSOVERS:
        child_chromosomes.append(crossover(population.pop(), population.pop()))
        spawned_children += 1
    return child_chromosomes


def age_population(population):
    for chromosome in population:
        chromosome.increment_age()
