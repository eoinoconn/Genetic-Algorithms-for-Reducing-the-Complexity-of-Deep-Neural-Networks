from Python.GeneticAlgorithm.mutate import create_parent, mutate
import operator
import logging

POOL_SIZE = 6


def get_best(max_generations):

    logger = logging.getLogger('geneticEngine')
    logger.info('Starting genetic engine...')

    generation = 1
    population = create_population(logger)
    logger.info("Generation number: %d", generation)
    while generation < max_generations:
        assess_population_fitness(population, logger)
        mutate_population(population, logger)
        generation += 1


def create_population(logger):
    pool = []
    for x in range(POOL_SIZE):
        pool.append(create_parent())
        logger.info("Added chromosome number %d to population", x+1)
    return pool


def assess_population_fitness(population, logger):
    i = 0
    for chromosome in population:
        logger.info("getting fitness of chromosome %d", i+1)
        chromosome.get_fitness()
        i += 1
    population.sort(key=operator.attrgetter('fitness'))


def mutate_population(population, logger):
    i = 0
    for chromosome in population:
        logger.info("mutating chromosome %d", i+1)
        mutate(chromosome.genes)
        i += 1
