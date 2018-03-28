import configparser, logging


class Node(object):

    def __init__(self):
        type = []


class ConvNode(Node):

    def __init__(self, random_initialisation=False):
        super().__init__()
        layer = []
        if random_initialisation:
            # initilise node randomly
            raise NotImplementedError
        else:
            # basic node initialisation
            raise NotImplementedError


class DenseNode(Node):
    def __init__(self, random_initialisation=False):
        super().__init__()
        self.id = 0
        self.layer = []
        if random_initialisation:
            # initilise node randomly
            raise NotImplementedError
        else:
            # basic node initialisation
            raise NotImplementedError


class Input(Node):
    def __init__(self, random_initialisation=False):
        super().__init__()
        self.id = 0
        self.layer = []
        if random_initialisation:
            # initilise node randomly
            raise NotImplementedError
        else:
            # basic node initialisation
            raise NotImplementedError


class Flatten(Node):
    def __init__(self, random_initialisation=False):
        super().__init__()
        self.id = 0
        self.layer = []
        if random_initialisation:
            # initilise node randomly
            raise NotImplementedError
        else:
            # basic node initialisation
            raise NotImplementedError


class Output(Node):
    def __init__(self, random_initialisation=False):
        super().__init__()
        self.id = 0
        self.layer = []
        if random_initialisation:
            # initilise node randomly
            raise NotImplementedError
        else:
            # basic node initialisation
            raise NotImplementedError


class Chromosome(object):

    def __init__(self):
        self.hyperparameters = []
        self.nodes = []
        self.vertices = []

        config = configparser.ConfigParser()
        config.read('GeneticAlgorithm/Config/training_parameters.ini')
        logger = logging.getLogger('Chromosome')
        logger.info("creating parent genes")

        if config['initial.generation'].getboolean('random_initial_generation'):
                raise NotImplementedError
                # random initial generation implementation
        else:
                raise NotImplementedError
                # the opposite

