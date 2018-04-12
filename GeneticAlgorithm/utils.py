from GeneticAlgorithm.genes import Genes
import configparser
import logging
import csv


def get_config():
    config = configparser.ConfigParser()
    config.read('GeneticAlgorithm/Config/training_parameters.ini')
    return config


def config_min_max_interval(config_name):
    config = get_config()
    config = config[config_name]
    minimum = int(config['minimum'])
    maximum = int(config['maximum'])
    interval = int(config['interval'])
    return minimum, maximum, interval


def load_known_architecture(file_name, input_shape):
    chromosome = Genes(input_shape)
    with open(file_name, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            line = line
            for j, x in enumerate(line):
                line[j] = convert(x)

            if i == 0:
                chromosome.hyperparameters = line
            else:
                chromosome.add_layer(line)
    return chromosome


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


def intermittent_logging(chromosome, generation_num):
    with open('GeneticAlgorithm/logs/trend.csv', 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow([generation_num, ',',
                             chromosome.id, ',',
                             chromosome.age, ',',
                             chromosome.accuracy, ',',
                             chromosome.fitness, ',',
                             chromosome.parameters, ',',
                             len(chromosome), ',',
                             chromosome.num_conv_nodes(), ',',
                             chromosome.num_dense_nodes(), ',',
                             ])


def setup_csvlogger():
    with open('GeneticAlgorithm/logs/trend.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['generation', ',',
                             'id', ',',
                             'Age', ',',
                             'Fitness', ',',
                             'Accuracy', ',',
                             'Parameters', ',',
                             'Num Layers', ',',
                             'Num Conv Layers', ',',
                             'Num Dense Layers', ',',
                             ])
