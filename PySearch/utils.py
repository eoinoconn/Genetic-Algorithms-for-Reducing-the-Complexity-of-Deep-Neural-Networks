from GeneticAlgorithm.genes import Genes
import configparser
import logging
import csv


def get_config():
    config = configparser.ConfigParser()
    config.read('PySearch/Config/training_parameters.ini')
    return config


def config_min_max_interval(config_name):
    config = get_config()
    config = config[config_name]
    minimum = int(config['minimum'])
    maximum = int(config['maximum'])
    interval = int(config['interval'])
    return minimum, maximum, interval



def convert(x):
    """
    Converts string to int, float or leaves  it as a string
    :param x:
    :return:
    """

    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return x


