import logging
import configparser
import csv

from keras.callbacks import EarlyStopping, History, TensorBoard
from keras.utils import print_summary

class Fitness(object):

    def __init__(self, efficiency_balance=0.000001):
        self.efficiency_balance = efficiency_balance
        self.epochs = 0
        self.batch_size = 0
        self.loss = 0
        self.optimizer = 0
        self.validation_split = 0
        self.verbose = 0
        self.hyper_params = []

        self.config = configparser.ConfigParser()
        self.config.read('PySearch/training_parameters.ini')

        # initilise logging objects
        self.logger_fitness = logging.getLogger('fitness')

        self._initialise_values()

    def _initialise_values(self):

        if self.config['training.parameters'].getboolean('overwrite_hyperparameters'):
            self.logger_fitness.info("Overwriting hyperparameters")
            self.efficiency_balance = float(self.config['training.parameters']['efficiency_balance'])
            self.epochs = int(self.config['training.parameters']['epochs'])
            self.batch_size = int(self.config['training.parameters']['batch_size'])
            self.loss = self.config['training.parameters']['loss_function']
            self.optimizer = self.config['training.parameters']['optimizer']
        else:
            self.epochs = self.hyper_params[2]
            self.batch_size = self.hyper_params[3]
            self.loss = self.hyper_params[0],
            self.optimizer = self.hyper_params[1]

        self.validation_split = float(self.config['training.parameters']['validation_split'])
        self.verbose = int(self.config['training.parameters']['fit_verbose'])

    def __call__(self, model, hyper_params,
                                  efficiency_balance=0.0000001,
                                  train_dataset=None, train_labels=None,
                                  valid_dataset=None, valid_labels=None,
                                  test_dataset=None, test_labels=None,
                                  evaluate_best=False, log_csv=False,
                                  eval_epochs=None):

        # log geneset model
        print_summary(model, print_fn=self.logger_fitness.info)

        self.logger_fitness.info("Model built successfully")

        if eval_epochs is not None:
            self.epochs = eval_epochs

        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=['accuracy'])

        self.logger_fitness.info("Model compiled succesfully, beginning training...")

        # get callbacks
        callbacks = self._get_callbacks()

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

        self.logger_fitness.info("Model trained successfully,  saving weights...")

        # store num of model parameters
        parameters = model.count_params()
        accuracy = hist.history['val_acc'][-1]

        self.logger_fitness.info("Weights saved")

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

        if self.config['efficient.cost.function'].getboolean('enable_efficiency_function'):
            fitness = self.cost_function(accuracy, self.efficiency_balance, parameters)
        else:
            self.logger_fitness.info("Fitness function disabled")
            fitness = accuracy

        self.logger_fitness.info("Model evaluated successfully, fitness = %.6f, accuracy = %.6f, parameters = %d\n\n",
                            fitness,
                            accuracy,
                            parameters)
        return fitness, accuracy, parameters

    @staticmethod
    def cost_function(accuracy, efficiency_balance, parameters):
        return accuracy - (efficiency_balance * parameters)


    def _get_callbacks(self):
        callbacks = []
        if self.config['early.stopping'].getboolean('early_stopping'):
            callbacks.append(EarlyStopping(monitor=self.config['early.stopping']['monitor'],
                                           min_delta=float(self.config['early.stopping']['min_delta']),
                                           patience=int(self.config['early.stopping']['patience']),
                                           verbose=int(self.config['early.stopping']['verbose']),
                                           mode=self.config['early.stopping']['mode']))

        if self.config['training.parameters'].getboolean('tensorboard'):
            callbacks.append(TensorBoard(log_dir='./Graph', histogram_freq=0,
                                         write_graph=True, write_images=True))
        return callbacks
