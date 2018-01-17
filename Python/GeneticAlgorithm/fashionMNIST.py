"""
In this file we will test the encoding and train a basic network using the built structure

"""
from __future__ import print_function
from Python.GeneticAlgorithm.fitness import *
from Python.GeneticAlgorithm.mutate import *
from Python.GeneticAlgorithm.genetic import *

import numpy as np
from six.moves import cPickle as pickle
import keras as keras
import unittest
import logging
from logging import config

image_size = 28
num_labels = 10


def reformat(dataset):
    dataset = np.expand_dims(dataset.reshape((-1, image_size, image_size)).astype(np.float32), axis=3)
    return dataset


def data_preprocess(logger=None, pickle_file='fashionMNIST.pickle'):
    logging.getLogger('resultMetrics').info("Data file: %s", pickle_file)
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save
        if logger is not None:
            logger.info('Training set', train_dataset.shape, train_labels.shape)
            logger.info('Test set', test_dataset.shape, test_labels.shape)

    train_dataset = reformat(train_dataset)
    test_dataset = reformat(test_dataset)

    train_labels = keras.utils.to_categorical(train_labels, num_labels)
    test_labels = keras.utils.to_categorical(test_labels, num_labels)

    return train_dataset, train_labels, test_dataset, test_labels


def get_fitness(logger=None, chromo=None, optimal_fitness=False):
    if optimal_fitness:
        return Fitness(optimal_fitness=True)
    train_dataset, train_labels, test_dataset, test_labels = data_preprocess(logger=logger)
    kwag = {"train_dataset": train_dataset, "train_labels": train_labels,
            "valid_dataset": None, "valid_labels": None,
            "test_dataset": test_dataset, "test_labels": test_labels}
    return Fitness(genes=chromo, **kwag)


class EncodingTest(unittest.TestCase):

    def test_encoding(self):
        logging.config.fileConfig('logging.conf')
        logger = logging.getLogger('testFile')

        def fnDisplay(candidate):
            logger.debug(candidate)

        def fnGetFitness(chromo):
            return get_fitness(chromo=chromo)

        def fnCustomMutate(chromo):
            return mutate(chromo)

        def fnCustomCreate():
            return create_parent()

        optimalFitness = get_fitness(optimal_fitness=True, logger=logger)
        best = get_best(fnGetFitness, None, optimalFitness, None, fnDisplay,
                        custom_mutate=fnCustomMutate, custom_create=fnCustomCreate, maxAge=50, max_generated_chromosomes=1000)
        self.assertTrue(not optimalFitness > best.Fitness)


if __name__ == '__main__':
    unittest.main()
