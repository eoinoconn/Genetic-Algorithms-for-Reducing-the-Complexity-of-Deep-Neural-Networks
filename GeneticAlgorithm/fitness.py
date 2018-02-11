import logging
import configparser

from keras.callbacks import EarlyStopping, History, TensorBoard
from keras.utils import print_summary


def assess_chromosome_fitness(genes, efficiency_balance=0.0000001,
                              train_dataset=None, train_labels=None,
                              valid_dataset=None, valid_labels=None,
                              test_dataset=None, test_labels=None,
                              evaluate_best=False):
    genes.log_geneset()

    config = configparser.ConfigParser()
    config.read('GeneticAlgorithm/Config/training_parameters.ini')

    # initilise logging objects
    logger_fitness = logging.getLogger('fitness')

    # log geneset id
    logger_fitness.info("Geneset id: %d\tAge: %d", genes.id, genes.age)

    # build model
    logger_fitness.info("building model")
    model = genes.build_model()

    # log geneset model
    print_summary(model, print_fn=logger_fitness.info)

    logger_fitness.info("Model built successfully, compiling...")

    # get hyperparameters
    hyper_params = genes.hyperparameters

    if config['training.parameters']['overwrite_hyperparameters']:
        efficiency_balance = float(config['training.parameters']['efficiency_balance'])

        model.compile(loss=config['training.parameters']['loss_function'],
                      optimizer=config['training.parameters']['optimizer'],
                      metrics=['accuracy'])
    else:
        model.compile(loss=hyper_params[0],
                      optimizer=hyper_params[1],
                      metrics=['accuracy'])

    logger_fitness.info("Model compiled succesfully, beginning training...")

    callbacks = []

    if config['early.stopping']['early_stopping'] == 'True':
        callbacks.append(EarlyStopping(monitor=config['early.stopping']['monitor'],
                                       min_delta=float(config['early.stopping']['min_delta']),
                                       patience=int(config['early.stopping']['patience']),
                                       verbose=int(config['early.stopping']['verbose']),
                                       mode=config['early.stopping']['mode']))

    if config['training.parameters']['tensorboard'] == 'True':
        callbacks.append(TensorBoard(log_dir='./Graph', histogram_freq=0,
                                     write_graph=True, write_images=True))

    if config['training.parameters']['overwrite_hyperparameters']:
        if valid_dataset is None:
            hist = model.fit(train_dataset, train_labels,
                             epochs=int(config['training.parameters']['epochs']),
                             batch_size=int(config['training.parameters']['batch_size']),
                             validation_split=float(config['training.parameters']['validation_split']),
                             callbacks=callbacks,
                             verbose=int(config['training.parameters']['fit_verbose']))

        else:
            hist = model.fit(train_dataset, train_labels,
                             epochs=int(config['training.parameters']['epochs']),
                             batch_size=int(config['training.parameters']['batch_size']),
                             validation_data=(valid_dataset, valid_labels),
                             callbacks=callbacks,
                             verbose=int(config['training.parameters']['fit_verbose']))
    else:
        if valid_dataset is None:
            hist = model.fit(train_dataset, train_labels,
                             epochs=hyper_params[2],
                             batch_size=hyper_params[3],
                             validation_split=0.16,
                             callbacks=callbacks,
                             verbose=0)

        else:
            hist = model.fit(train_dataset, train_labels,
                             epochs=hyper_params[2],
                             batch_size=hyper_params[3],
                             validation_data=(valid_dataset, valid_labels),
                             callbacks=callbacks,
                             verbose=0)

    logger_fitness.info("Model trained succesfully, beginning evaluation...")

    # store num of model parameters
    parameters = model.count_params()
    accuracy = hist.history['val_acc'][-1]

    if evaluate_best:
        if config['training.parameters']['overwrite_hyperparameters']:
            loss_and_metrics = model.evaluate(test_dataset, test_labels,
                                              batch_size=int(config['training.parameters']['batch_size']),
                                              verbose=int(config['training.parameters']['evaluate_verbose']))
        else:
            loss_and_metrics = model.evaluate(test_dataset, test_labels,
                                              batch_size=hyper_params[3],
                                              verbose=0)
        accuracy = loss_and_metrics[1]

    fitness = cost_function(accuracy, efficiency_balance, parameters)

    logger_fitness.info("Model evaluated successfully, fitness = %.6f, accuracy = %.6f, parameters = %d",
                        fitness,
                        accuracy,
                        parameters)
    return fitness, accuracy, parameters


def cost_function(accuracy, efficiency_balance, parameters):
    return accuracy - (efficiency_balance * parameters)
