import logging
import configparser
import csv

from keras.callbacks import EarlyStopping, History, TensorBoard
from keras.utils import print_summary


def assess_chromosome_fitness(model, hyper_params,
                              efficiency_balance=0.0000001,
                              train_dataset=None, train_labels=None,
                              valid_dataset=None, valid_labels=None,
                              test_dataset=None, test_labels=None,
                              evaluate_best=False, log_csv=False,
                              eval_epochs=None):
    config = configparser.ConfigParser()
    config.read('../GeneticAlgorithm/Config/training_parameters.ini')

    # initilise logging objects
    logger_fitness = logging.getLogger('fitness')

    # log geneset model
    print_summary(model, print_fn=logger_fitness.info)

    logger_fitness.info("Model built successfully")

    if config['training.parameters'].getboolean('overwrite_hyperparameters'):
        logger_fitness.info("Overwriting hyperparameters")
        efficiency_balance = float(config['training.parameters']['efficiency_balance'])
        epochs = int(config['training.parameters']['epochs'])
        batch_size = int(config['training.parameters']['batch_size'])
        loss = config['training.parameters']['loss_function']
        optimizer = config['training.parameters']['optimizer']
    else:
        epochs = hyper_params[2]
        batch_size = hyper_params[3]
        loss = hyper_params[0],
        optimizer = hyper_params[1]

    if eval_epochs is not None:
        epochs = eval_epochs

    validation_split = float(config['training.parameters']['validation_split'])
    verbose = int(config['training.parameters']['fit_verbose'])

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    logger_fitness.info("Model compiled succesfully, beginning training...")

    # get callbacks
    callbacks = get_callbacks(config)

    if valid_dataset is None:
        hist = model.fit(train_dataset, train_labels,
                         epochs=epochs,
                         batch_size=batch_size,
                         # steps_per_epoch = train_dataset.shape[0] // batch_size,
                         validation_split=validation_split,
                         callbacks=callbacks,
                         verbose=verbose)

    else:
        hist = model.fit(train_dataset, train_labels,
                         epochs=epochs,
                         #steps_per_epoch=32,
                         batch_size=batch_size,
                         validation_data=(valid_dataset, valid_labels),
                         callbacks=callbacks,
                         verbose=verbose)

    logger_fitness.info("Model trained successfully,  saving weights...")

    # store num of model parameters
    parameters = model.count_params()
    accuracy = hist.history['val_acc'][-1]

    logger_fitness.info("Weights saved")

    if evaluate_best:
        loss_and_metrics = model.evaluate(test_dataset, test_labels,
                                          batch_size=batch_size,
                                          verbose=verbose)

        if log_csv:
            with open('GeneticAlgorithm/logs/validvtest.csv', 'a', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=' ',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow([accuracy, ',', loss_and_metrics[1]])
        accuracy = loss_and_metrics[1]

    if config['efficient.cost.function'].getboolean('enable_efficiency_function'):
        fitness = cost_function(accuracy, efficiency_balance, parameters)
    else:
        logger_fitness.info("Fitness function disabled")
        fitness = accuracy

    logger_fitness.info("Model evaluated successfully, fitness = %.6f, accuracy = %.6f, parameters = %d\n\n",
                        fitness,
                        accuracy,
                        parameters)
    return fitness, accuracy, parameters


def cost_function(accuracy, efficiency_balance, parameters):
    return accuracy - (efficiency_balance * parameters)


def get_callbacks(config):
    callbacks = []
    if config['early.stopping'].getboolean('early_stopping'):
        callbacks.append(EarlyStopping(monitor=config['early.stopping']['monitor'],
                                       min_delta=float(config['early.stopping']['min_delta']),
                                       patience=int(config['early.stopping']['patience']),
                                       verbose=int(config['early.stopping']['verbose']),
                                       mode=config['early.stopping']['mode']))

    if config['training.parameters'].getboolean('tensorboard'):
        callbacks.append(TensorBoard(log_dir='./Graph', histogram_freq=0,
                                     write_graph=True, write_images=True))
    return callbacks
