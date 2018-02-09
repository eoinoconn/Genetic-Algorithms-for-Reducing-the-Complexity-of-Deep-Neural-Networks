from keras.utils import to_categorical

from GeneticAlgorithm.crossover import crossover
from GeneticAlgorithm.mutate import create_parent, mutate
from keras.datasets import cifar10, mnist
import operator
import logging
import random
from keras import backend as K
import io
import csv
import numpy as np

POOL_SIZE = 10
random.seed(1994)

NUM_LABELS = 10
MAX_CROSSOVERS = 4


def get_best(max_generations, input_shape, fn_unpack_training_data):

    logger = logging.getLogger('geneticEngine')
    logger.info('Starting genetic engine...')

    setup_csvlogger()

    training_data = fn_unpack_training_data()

    generation = 1
    population = create_population(input_shape, logger)
    best_chromosome = population[0]

    while generation < max_generations:
        logger.info("Generation number: %d", generation)
        logger.info("pool size: %d", population.__len__())

        # mutate pool
        mutate_population(population, logger)

        # Assign population fitness
        best_child = assess_population_fitness(population, training_data, logger)

        # if new best chromosome found, save it
        if best_child > best_chromosome:
            best_chromosome = best_child
            best_chromosome.log_best()

        intermitent_logging(best_child)

        # select best chromosomes
        population.extend(spawn_children(population, input_shape, logger))

        # iterate the age of every chromosome in the population by 1
        age_population(population)

        logger.info("End of generation %d \n\n", generation)
        generation += 1
    return best_chromosome


def create_population(input_shape, logger):
    pool = []
    for x in range(POOL_SIZE):
        pool.append(create_parent(input_shape))
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
    random.shuffle(population)
    for chromosome in population:
        logger.info("mutating chromosome %d", chromosome.id)
        mutate(chromosome)
        mutations_completed += 1


def spawn_children(population, input_shape, logger):
    child_chromosomes = []
    spawned_children = 0
    while population.__len__() > 1 and spawned_children < MAX_CROSSOVERS:
        parent_1 = population.pop()
        parent_2 = population.pop()
        logger.info("Spawning children from chromosomes %d and %d", parent_1.id, parent_2.id)
        child_chromosomes.append(crossover(parent_1, parent_2, input_shape))
        child_chromosomes.append(crossover(parent_2, parent_1, input_shape))
        spawned_children += 2
    return child_chromosomes


def age_population(population):
    for chromosome in population:
        chromosome.increment_age()


def intermitent_logging(chromosome):
    with open('GeneticAlgorithm/logs/trend.csv', 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow([chromosome.id, ',', chromosome.age, ',', chromosome.accuracy, ',',
                             chromosome.fitness, ',', chromosome.parameters])


def setup_csvlogger():
    with open('GeneticAlgorithm/logs/trend.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow([0])
