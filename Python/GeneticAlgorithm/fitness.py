import logging
from keras.utils import print_summary
from keras.callbacks import EarlyStopping


class Fitness:
    def __init__(self, optimal_fitness=False, genes=None, beta=0.0000001,
                 train_dataset=None, train_labels=None,
                 valid_dataset=None, valid_labels=None,
                 test_dataset=None, test_labels=None):

        self.beta = beta

        if optimal_fitness:
            self.accuracy = 0.98
            self.parameters = 100000

            # log setup variable
            logger_fitness = logging.getLogger('fitness')
            logger_fitness.info('Beta: %4.2f', self.fitness())

        else:

            self.genes = genes

            # initilise logging objects
            logger_fitness = logging.getLogger('fitness')
            logger_genes = logging.getLogger('geneset')

            # log geneset id
            logger_fitness.info("Geneset id: %d", genes.id)
            logger_genes.info("Geneset id: %d", genes.id)

            # log gene shape
            logger_genes.info(genes.__str__())

            # build model
            logger_fitness.info("building model")
            self.model = genes.build_model()

            # log geneset model
            print_summary(self.model, print_fn=logger_fitness.info)
            print_summary(self.model, print_fn=logger_genes.info)

            logger_fitness.info("Model built successfully, compiling...")

            # get hyperparameters
            hyper_params = genes.hyperparameters

            self.model.compile(loss=hyper_params[0],
                               optimizer=hyper_params[1],
                               metrics=['accuracy'], )

            logger_fitness.info("Model compiled succesfully, beginning training")

            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.005, patience=2, verbose=0, mode='auto')

            self.model.fit(train_dataset, train_labels,
                           epochs=hyper_params[2],
                           batch_size=hyper_params[3],
                           validation_data=(valid_dataset, valid_labels),
                           callbacks=[early_stopping],
                           verbose=2)
            loss_and_metrics = self.model.evaluate(test_dataset, test_labels,
                                                   batch_size=hyper_params[3],
                                                   verbose=0)

            # store num of model parameters
            self.parameters = self.model.count_params()

            self.accuracy = loss_and_metrics[1]
            logger_fitness.info("Model trained succesfully, fitness = %4.2f, accuracy = %.2f, parameters = %d",
                                self.fitness(),
                                self.accuracy, self.parameters)

    def fitness(self):
        return self.accuracy - (self.beta * self.parameters)
        
    def new_best(self, age):
        logger = logging.getLogger('resultMetrics')
        logger.info("new best genes, id = %d, age = %d", self.genes.id, age)
        print_summary(self.model, print_fn=logger.info)
        logger.info("Fitness: %4.2f\tAccuracy: %6.4f\tParameters %d\n", self.fitness(), self.accuracy, self.parameters)

    def __str__(self):
        return "Fitness: {:4.2f}\tAccuracy: {:4.2f}\tParameters: {:4.2f}\n".format(
            self.fitness(),
            self.accuracy,
            self.parameters

        )

    def __gt__(self, other):
        return self.fitness() > other.fitness()
