"""
In this file we will test the encoding and train a basic network using the built structure

"""
from __future__ import print_function
import time

import sys

from GeneticAlgorithm.fitness import assess_chromosome_fitness
from GeneticAlgorithm.genetic_engine import *

import subprocess
import unittest
import logging
from logging import config

image_size = 28
num_labels = 10

img_rows = 28
img_cols = 28


def unpack_testing_data(num_labels):
    (train_dataset, train_labels), (test_dataset, test_labels) = fashion_mnist.load_data()

    if K.image_data_format() == 'channels_first':
        train_dataset = train_dataset.reshape(train_dataset.shape[0], 1, img_rows, img_cols)
        test_dataset = test_dataset.reshape(test_dataset.shape[0], 1, img_rows, img_cols)
    else:
        train_dataset = train_dataset.reshape(train_dataset.shape[0], img_rows, img_cols, 1)
        test_dataset = test_dataset.reshape(test_dataset.shape[0], img_rows, img_cols, 1)

    train_dataset = train_dataset.astype('float32')
    test_dataset = test_dataset.astype('float32')
    train_dataset /= 255
    test_dataset /= 255

    train_labels = to_categorical(train_labels, num_labels)
    test_labels = to_categorical(test_labels, num_labels)

    return {"train_dataset": train_dataset, "train_labels": train_labels,
            "valid_dataset": None, "valid_labels": None,
            "test_dataset": test_dataset, "test_labels": test_labels}


def get_git_hash():
    return subprocess.check_output(["git", "describe", "--always"]).strip()


def set_seed(logger):
    seed = random.randrange(sys.maxsize)
    rng = random.Random(seed)
    logger.info("Seed was: %f", seed)


class MNISTTest(unittest.TestCase):

    def test_encoding(self):
        logging.config.fileConfig('GeneticAlgorithm/logs/logging.conf')
        logger = logging.getLogger('testFile')

        logger.info("Setting seed")
        set_seed(logger)

        logger.info("starting test...")

        logger.info(get_git_hash())

        start = time.time()
        best = get_best(3, (28, 28, 1), unpack_testing_data(10))
        end = time.time()

        logger.info("time to best %f", end-start)
        self.assertTrue(assess_chromosome_fitness(best, evaluate_best=True, eval_epochs=100, **unpack_testing_data(10))[0] > 0.99)


if __name__ == '__main__':
    unittest.main()
