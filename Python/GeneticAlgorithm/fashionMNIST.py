"""
In this file we will test the encoding and train a basic network using the built structure

"""
from __future__ import print_function
from Python.GeneticAlgorithm.fitness import *
from Python.GeneticAlgorithm.mutate import *
from Python.GeneticAlgorithm.genetic import *

from keras.datasets import fashion_mnist
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

    from keras.datasets import fashion_mnist

    (train_dataset, train_labels), (test_dataset, test_labels) = fashion_mnist.load_data()

    train_dataset = train_dataset[:60000]
    train_labels = train_labels[:60000]
    test_dataset = test_dataset[:10000]
    test_labels = test_labels[:10000]

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

        logging.getLogger('resultMetrics').info("Data file: fashionMNIST")

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
