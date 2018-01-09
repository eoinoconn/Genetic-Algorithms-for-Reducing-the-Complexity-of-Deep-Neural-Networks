import logging
from keras.utils import print_summary


class Fitness:
    def __init__(self, optimal_fitness=False, genes=None, train_dataset=None, train_labels=None,
                 valid_dataset=None,
                 valid_labels=None, test_dataset=None, test_labels=None):
        if optimal_fitness:
            self.accuracy = 0.90
        else:
            # initilise logging objects
            logger_fitness = logging.getLogger('geneset')
            logger_genes = logging.getLogger('genes')

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

            logger_fitness.info("Model built succesfully, compiling...")

            self.model.compile(loss='categorical_crossentropy',
                               optimizer='sgd',
                               metrics=['accuracy'], )

            logger_fitness.info("Model compiled succesfully, beginning training")

            self.model.fit(train_dataset, train_labels,
                           epochs=1,
                           batch_size=100,
                           validation_data=(valid_dataset, valid_labels),
                           verbose=2)
            loss_and_metrics = self.model.evaluate(test_dataset, test_labels,
                                                   batch_size=100,
                                                   verbose=0)

            self.accuracy = loss_and_metrics[1]
            logger_fitness.info("Model trained succesfully, accuracy = %.2f", self.accuracy)

    def new_best(self):
        logger = logging.getLogger('resultMetrics')
        logger.info("new best genes")
        print_summary(self.model, print_fn=logger.info)
        logger.info("Accuracy: %4.2f\tParameters %d\n", self.accuracy, self.model.count_params())

    def __str__(self):
        return "{} Accuracy\n".format(
            self.accuracy
        )

    def __gt__(self, other):
        return self.accuracy > (other.accuracy - 0.01)
