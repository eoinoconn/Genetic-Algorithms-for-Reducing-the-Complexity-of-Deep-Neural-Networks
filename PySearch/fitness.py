import logging
import configparser
import csv

from keras.callbacks import EarlyStopping, History, TensorBoard
from keras.utils import print_summary

class Fitness(object):
    """
    Fitness object to evaluate chromosome.
    
    Arguments:
        efficiency_balance -- scaling factor for fitness function (Default = 0.0000001).
    """
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
        """
        Initialise training parameters.
        Overwrites chromosome hyperparameters if 
        training.parameters.overwrite_hyperparameters = True
        """
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
                                  train_dataset=None, train_labels=None,
                                  valid_dataset=None, valid_labels=None,
                                  test_dataset=None, test_labels=None,
                                  eval_epochs=None):
        """
        Trains model and evaulates using validation data or test data.
        
        Arguments:
        
            model -- the keras model of the chromosome.
            hyper_params -- list of chromosome model hyper-parameters.
            train_dataset -- numpy array of trainning data.
            train_labels -- numpy array of trainning data labels.
            valid_dataset --  numpy array of validation data.
            valid_labels -- numpy array of validation data labels.
            test_dataset --  numpy array of testing data.
            test_labels -- numpy array of testing data.
            evaluate_best -- if true, evaluates on test data (Default=False).
            eval_epochs -- if not None, overwrites number of training epochs (Default=None).
        
        Returns:

            Fitness -- [0,1] Fitness of the evaluated chromosome. If efficient.cost.function is False
                Fitness is assigned the value of accuracy.
            Accuracy -- [0,1] Accuracy of evaluated network.
            Parameters -- [0, inf] Number of trainable parameters a network has.
        """
        # log geneset model
        print_summary(model, print_fn=self.logger_fitness.info)

        if eval_epochs is not None:
            self.epochs = eval_epochs

        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=['accuracy'])

        self.logger_fitness.info("Model compiled succesfully, beginning training...")

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

        self.logger_fitness.info("Model trained successfully")

        parameters = model.count_params()
        accuracy = hist.history['val_acc'][-1]

        # if test dataset present evalualte using test data.
        if test_dataset is not None:
            loss_and_metrics = model.evaluate(test_dataset, test_labels,
                                              batch_size=self.batch_size,
                                              verbose=self.verbose)
            accuracy = loss_and_metrics[1]

        # if fitness function is active utilise.
        if self.config['efficient.cost.function'].getboolean('enable_efficiency_function'):
            fitness = self.cost_function(accuracy, self.efficiency_balance, parameters)
        # else assign accuracy as fitness
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
        """
        Returns result of fitness function.
        :return: float
        """
        return accuracy - (efficiency_balance * parameters)


    def _get_callbacks(self):
        """
        Initilises appropriate callbacks for training.

        Callbacks include Early stopping and TensorBoard, with parameters set by training_parameters.ini
        :return: List of callback objects
        """
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
