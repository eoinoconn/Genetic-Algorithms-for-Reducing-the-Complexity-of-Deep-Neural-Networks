import logging
import configparser
import csv

from keras.callbacks import EarlyStopping, History, TensorBoard
from keras.utils import print_summary

class Fitness

    def __int__(self, efficiency_balance):
        self.efficiency_balance = efficiency_balance
        self.epochs = 0
        self.batch_size = 0
        self.loss = 0
        self.optimizer = 0
        self.validation_split = 0
        self.verbose = 0

        config = configparser.ConfigParser()
        config.read('GeneticAlgorithm/Config/training_parameters.ini')

        # initilise logging objects
        logger_fitness = logging.getLogger('fitness')

        self.initialise_values()

    def initialise_values(self):

        if config['training.parameters'].getboolean('overwrite_hyperparameters'):
            self.logger.info("Overwriting hyperparameters")
            self.efficiency_balance = float(config['training.parameters']['efficiency_balance'])
            self.epochs = int(config['training.parameters']['epochs'])
            self.batch_size = int(config['training.parameters']['batch_size'])
            self.loss = config['training.parameters']['loss_function']
            self.optimizer = config['training.parameters']['optimizer']
        else:
            self.epochs = hyper_params[2]
            self.batch_size = hyper_params[3]
            self.loss = hyper_params[0],
            self.optimizer = hyper_params[1]

        self.validation_split = float(config['training.parameters']['validation_split'])
        self.verbose = int(config['training.parameters']['fit_verbose'])

    def assess_chromosome_fitness(model, hyper_params,
                                  efficiency_balance=0.0000001,
                                  train_dataset=None, train_labels=None,
                                  valid_dataset=None, valid_labels=None,
                                  test_dataset=None, test_labels=None,
                                  evaluate_best=False, log_csv=False,
                                  eval_epochs=None):

        # log geneset model
        print_summary(model, print_fn=self.logger.info)

        logger_fitness.info("Model built successfully")

        if eval_epochs is not None:
            self.epochs = eval_epochs

        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=['accuracy'])

        logger_fitness.info("Model compiled succesfully, beginning training...")

        # get callbacks
        callbacks = self.get_callbacks(config)

        if valid_dataset is None:
            hist = model.fit(train_dataset, train_labels,
                             epochs=self.epochs,
                             batch_size=self.batch_size,
                             validation_split=self.validation_split,
                             callbacks=callbacks,
                             verbose=self.verbose)

        else:
            hist = model.fit(train_dataset, train_labels,
                             epochs=self.epochs,
                             batch_size=self.batch_size,
                             validation_data=(valid_dataset, valid_labels),
                             callbacks=callbacks,
                             verbose=self.verbose)

        logger_fitness.info("Model trained successfully,  saving weights...")

        # store num of model parameters
        parameters = model.count_params()
        accuracy = hist.history['val_acc'][-1]

        logger_fitness.info("Weights saved")

        if evaluate_best:
            loss_and_metrics = model.evaluate(test_dataset, test_labels,
                                              batch_size=self.batch_size,
                                              verbose=self.verbose)

            if log_csv:
                with open('GeneticAlgorithm/logs/validvtest.csv', 'a', newline='') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=' ',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    spamwriter.writerow([accuracy, ',', loss_and_metrics[1]])
            accuracy = loss_and_metrics[1]

        if config['efficient.cost.function'].getboolean('enable_efficiency_function'):
            fitness = cost_function(accuracy, self.efficiency_balance, parameters)
        else:
            logger_fitness.info("Fitness function disabled")
            fitness = accuracy

        self.logger.info("Model evaluated successfully, fitness = %.6f, accuracy = %.6f, parameters = %d\n\n",
                            fitness,
                            accuracy,
                            parameters)
        return fitness, accuracy, parameters

    @staticmethod
    def cost_function(accuracy, efficiency_balance, parameters):
        return accuracy - (efficiency_balance * parameters)


    def get_callbacks(self):
        callbacks = []
        if self.config['early.stopping'].getboolean('early_stopping'):
            callbacks.append(EarlyStopping(monitor=config['early.stopping']['monitor'],
                                           min_delta=float(config['early.stopping']['min_delta']),
                                           patience=int(config['early.stopping']['patience']),
                                           verbose=int(config['early.stopping']['verbose']),
                                           mode=config['early.stopping']['mode']))

        if config['training.parameters'].getboolean('tensorboard'):
            callbacks.append(TensorBoard(log_dir='./Graph', histogram_freq=0,
                                         write_graph=True, write_images=True))
        return callbacks
