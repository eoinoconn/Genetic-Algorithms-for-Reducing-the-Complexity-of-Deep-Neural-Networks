"""
In this file we will test the encoding and train a basic network using the built structure

"""
from __future__ import print_function
import time
import random
import sys
import subprocess
import unittest
import logging
from PySearch import PySearch
from keras.datasets import cifar10
from logging import config


def get_git_hash():
    return subprocess.check_output(["git", "describe", "--always"]).strip()


def set_seed(logger):
    seed = random.randrange(sys.maxsize)
    random.Random(seed)
    logger.info("Seed was: %f", seed)


class CIFAR10Test(unittest.TestCase):

    def test_encoding(self):
        logging.config.fileConfig('PySearch/logs/logging.conf')
        logger = logging.getLogger('testFile')

        logger.info("Setting seed")
        set_seed(logger)

        logger.info("starting test...")

        logger.info(get_git_hash())

        (train_dataset, train_labels), (test_dataset,
                                        test_labels) = cifar10.load_data()

        start = time.time()
        best_cnn = PySearch(10, 
                            {'train_dataset' : train_dataset, 
                            'train_labels': train_labels,
                            'test_dataset': test_dataset,
                            'test_labels' : test_labels})
        fitness, accuracy, parameters = best_cnn(50, 0.00001)
        end = time.time()
        print("Time to best %f, fitness: %f, Accuracy: %f, Parameters: %d",
              end - start, fitness, accuracy, parameters)


if __name__ == '__main__':
    unittest.main()
