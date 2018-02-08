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

num_labels = 10
img_rows = 32
img_cols = 32


def unpack_training_data():
    (train_dataset, train_labels), (test_dataset, test_labels) = cifar10.load_data()

    if K.image_data_format() == 'channels_first':
        train_dataset = train_dataset.reshape(train_dataset.shape[0], 3, img_rows, img_cols)

    else:
        train_dataset = train_dataset.reshape(train_dataset.shape[0], img_rows, img_cols, 3)

    train_labels = to_categorical(train_labels, NUM_LABELS)

    train_dataset = train_dataset.astype('float32')
    train_dataset /= 255

    return {"train_dataset": train_dataset, "train_labels": train_labels,
            "valid_dataset": None, "valid_labels": None}


def unpack_testing_data():
    (train_dataset, train_labels), (test_dataset, test_labels) = cifar10.load_data()

    print('train_dataset shape:', train_dataset.shape)
    print(train_dataset.shape[0], 'train samples')
    print(test_dataset.shape[0], 'test samples')

    if K.image_data_format() == 'channels_first':
        train_dataset = train_dataset.reshape(train_dataset.shape[0], 3, img_rows, img_cols)
        test_dataset = test_dataset.reshape(test_dataset.shape[0], 3, img_rows, img_cols)
    else:
        train_dataset = train_dataset.reshape(train_dataset.shape[0], img_rows, img_cols, 3)
        test_dataset = test_dataset.reshape(test_dataset.shape[0], img_rows, img_cols, 3)

    train_labels = to_categorical(train_labels, NUM_LABELS)
    test_labels = to_categorical(test_labels, NUM_LABELS)

    train_dataset = train_dataset.astype('float32')
    test_dataset = test_dataset.astype('float32')
    train_dataset /= 255
    test_dataset /= 255

    return {"train_dataset": train_dataset, "train_labels": train_labels,
            "valid_dataset": None, "valid_labels": None,
            "test_dataset": test_dataset, "test_labels": test_labels}


class CIFAR10Test(unittest.TestCase):

    def test_encoding(self):
        logging.config.fileConfig('logs/logging.conf')
        logger = logging.getLogger('testFile')

        logger.info("starting test...")

        start = time.time()
        best = get_best(100, (32, 32, 3), unpack_training_data)
        end = time.time()

        logger.info("time to best %f", end-start)
        self.assertTrue(evaluate_best_chromosome(best, **unpack_testing_data()) > 0.99)


if __name__ == '__main__':
    unittest.main()
