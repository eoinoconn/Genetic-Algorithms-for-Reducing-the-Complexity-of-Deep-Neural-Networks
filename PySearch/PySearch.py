import operator
import logging
import configparser
from keras.utils import to_categorical

from PySearch.chromosome import Chromosome


class PySearch(object):

    def __init__(self, num_labels,
                 train_dataset=None, train_labels=None,
                 valid_dataset=None, valid_labels=None,
                 test_dataset=None, test_labels=None):

        self.config = configparser.ConfigParser()
        self.config.read('PySearch/training_parameters.ini')

        self.logger = logging.getLogger("genetic_engine")
        self.pool = []

        train_labels = to_categorical(train_labels, num_labels)
        test_labels = to_categorical(test_labels, num_labels)

        # Normalise training data
        train_dataset = train_dataset.astype('float32')
        test_dataset = test_dataset.astype('float32')
        train_dataset /= 255
        test_dataset /= 255

        self.training_data = {'train_dataset': train_dataset, 'train_labels': train_labels,
                              'valid_dataset': valid_dataset, 'valid_labels': valid_labels,
                              'test_dataset': test_dataset, 'test_labels': test_labels}

        self.input_size = train_dataset.shape[1:]
        self.logger.info("Dataset input size: %s", str(self.input_size))

        self.pool_size = int(self.config['genetic.engine']['pool_size'])
        self.logger.info("Pool size: %d", self.pool_size)

        self.max_crossovers = int(self.config['genetic.engine']['max_crossover'])
        self.logger.info("Generation Crossovers: %d", self.max_crossovers)


    def __call__(self, generations, factor):

        self._initialise_population(factor)

        current_best = self.pool[-1]

        for i in range(generations):

            self._assess_population()

            if self.pool[-1].fitness > current_best.fitness:
                current_best = self.pool[-1]
                current_best.assess(log="result")

            self._crossover_population()

            self._mutate_population()

        return current_best.assess(log="result")

    def _assess_population(self):
        for chromosome in self.pool:
            chromosome.evaluate(self.training_data)
        self.pool.sort(key=operator.attrgetter('fitness'))

    def _mutate_population(self):
        for chromosome in self.pool:
            chromosome.mutate()

    def _initialise_population(self, factor):
        for i in range(self.pool_size):
            self.pool.append(Chromosome(self.input_size))

    def _crossover_population(self):
        pass
