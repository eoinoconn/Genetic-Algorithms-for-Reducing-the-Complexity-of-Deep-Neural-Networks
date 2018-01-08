import logging
from keras.utils import print_summary


class Fitness:
    def __init__(self, optimal_fitness=False, chromosome=None, train_dataset=None, train_labels=None, valid_dataset=None,
                 valid_labels=None, test_dataset=None, test_labels=None):
        if optimal_fitness:
            self.accuracy = 0.90
        else:
            logger = logging.getLogger('fitness')
            logger.info("building model")
            model = chromosome.build_model()
            print_summary(model, print_fn=logger.debug)
            logger.info("Model built succesfully, compiling...")
            model.compile(loss='categorical_crossentropy',
                          optimizer='sgd',
                          metrics=['accuracy'],)
            logger.info("Model compiled succesfully, beginning training")
            model.fit(train_dataset, train_labels,
                      epochs=1,
                      batch_size=100,
                      validation_data=(valid_dataset, valid_labels),
                      verbose=2,)
            loss_and_metrics = model.evaluate(test_dataset, test_labels,
                                              batch_size=100,
                                              verbose=0)
            self.accuracy = loss_and_metrics[1]
            logger.info("Model trained succesfully, accuracy = %.2f", self.accuracy)

    def __str__(self):
        return "{} Accuracy\n".format(
            self.accuracy
        )

    def __gt__(self, other):
        return self.accuracy > (other.accuracy-0.01)
