"""
In this file we will test the encoding and train a basic network using the built structure

"""
from __future__ import print_function
import time

from Python.GeneticAlgorithm.fitness import assess_chromosome_fitness, evaluate_best_chromosome
from Python.GeneticAlgorithm.genetic_engine import *


import unittest
import logging
from logging import config

image_size = 28
num_labels = 10


def unpack_training_data():
    (train_dataset, train_labels), (test_dataset, test_labels) = mnist.load_data()

    train_dataset = reformat(train_dataset)
    test_dataset = reformat(test_dataset)

    train_labels = to_categorical(train_labels, NUM_LABELS)
    test_labels = to_categorical(test_labels, NUM_LABELS)

    return {"train_dataset": train_dataset, "train_labels": train_labels,
            "valid_dataset": None, "valid_labels": None}


def unpack_testing_data():
    (train_dataset, train_labels), (test_dataset, test_labels) = mnist.load_data()

    train_dataset = reformat(train_dataset)
    test_dataset = reformat(test_dataset)

    train_labels = to_categorical(train_labels, NUM_LABELS)
    test_labels = to_categorical(test_labels, NUM_LABELS)

    return {"train_dataset": train_dataset, "train_labels": train_labels,
            "valid_dataset": None, "valid_labels": None,
            "test_dataset": test_dataset, "test_labels": test_labels}


def reformat(dataset):
    dataset = np.expand_dims(dataset.reshape((-1, IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32), axis=3)
    return dataset


class MNISTTest(unittest.TestCase):

    def test_encoding(self):
        logging.config.fileConfig('logs/logging.conf')
        logger = logging.getLogger('testFile')

        logger.info("starting test...")

        start = time.time()
        best = get_best(100, unpack_training_data)
        end = time.time()

        logger.info("time to best %f", end-start)
        self.assertTrue(evaluate_best_chromosome(best, **unpack_testing_data()) > 0.99)


if __name__ == '__main__':
    unittest.main()
