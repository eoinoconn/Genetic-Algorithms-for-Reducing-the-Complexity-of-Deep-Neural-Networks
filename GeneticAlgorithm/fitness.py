import logging
import configparser
import csv

from keras.callbacks import EarlyStopping, History, TensorBoard
from keras.utils import print_summary


def assess_chromosome_fitness(genes, efficiency_balance=0.0000001,
                              train_dataset=None, train_labels=None,
                              valid_dataset=None, valid_labels=None,
                              test_dataset=None, test_labels=None,
                              evaluate_best=False, log_csv=False):

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

    logger_fitness.info("Model built successfully, restoring weights...")

    reuse_previous_weights(genes, model, logger_fitness)

    # get hyperparameters
    hyper_params = genes.hyperparameters

    if config['training.parameters'].getboolean('overwrite_hyperparameters'):
        logger_fitness.info("Overwriting hyperparameters")
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

    if config['early.stopping'].getboolean('early_stopping'):
        callbacks.append(EarlyStopping(monitor=config['early.stopping']['monitor'],
                                       min_delta=float(config['early.stopping']['min_delta']),
                                       patience=int(config['early.stopping']['patience']),
                                       verbose=int(config['early.stopping']['verbose']),
                                       mode=config['early.stopping']['mode']))

    if config['training.parameters'].getboolean('tensorboard'):
        callbacks.append(TensorBoard(log_dir='./Graph', histogram_freq=0,
                                     write_graph=True, write_images=True))

    if config['training.parameters'].getboolean('overwrite_hyperparameters'):
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

    logger_fitness.info("Model trained successfully,  saving weights...")

    # store num of model parameters
    parameters = model.count_params()
    accuracy = hist.history['val_acc'][-1]

    genes = save_model_weights(genes, model, logger_fitness)

    logger_fitness.info("Weights saved, evaluating model...")

    if evaluate_best:
        if config['training.parameters'].getboolean('overwrite_hyperparameters'):
            loss_and_metrics = model.evaluate(test_dataset, test_labels,
                                              batch_size=int(config['training.parameters']['batch_size']),
                                              verbose=int(config['training.parameters']['evaluate_verbose']))
        else:
            loss_and_metrics = model.evaluate(test_dataset, test_labels,
                                              batch_size=hyper_params[3],
                                              verbose=0)
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


def reuse_previous_weights(genes, model, logger):
    model_buffer = 1
    for i in range(genes.__len__()):
        weights_and_biases = genes.get_layer_weights(i)
        if weights_and_biases != 0:
            logger.info("re-using weights for layer %d", i)
            layer = genes.get_layer(i)
            if layer[0] == 1:
                model.layers[i+model_buffer].set_weights(layer[-1])
                model_buffer += 2   # increase buffer for dropout and activation
            elif layer[0] == 2:
                model.layers[i + model_buffer].set_weights(layer[-1])
                if layer[2] > 0:    # batch normalisation
                    model_buffer += 1
                    model.layers[i + model_buffer].set_weights(layer[-1])
                model_buffer += 1   # activation
                if layer[5] > 0:
                    model_buffer += 1
            elif layer[0] == 3:
                continue
            elif layer[0] == 4:
                inception_weights_and_biases = genes.get_layer_weights(i)
                for weight_and_bias in inception_weights_and_biases:
                    model.layers[i + model_buffer].set_weights(weight_and_bias)
                    model_buffer += 1
                model_buffer += 1   # concatenation layer
            else:
                raise NotImplementedError


def save_model_weights(genes, model, logger):
    model_buffer = 1
    for i in range(genes.__len__()):
        weights_and_biases = model.layers[i + model_buffer].get_weights()
        layer = genes.get_layer(i)
        if layer[0] == 1:
            layer[-1] = weights_and_biases
            model_buffer += 2   # dropout and activation
        elif layer[0] == 2:
            layer[-1] = weights_and_biases
            if layer[10] > 0:    # batch normalisation
                model_buffer += 1
                layer[-2] = model.layers[i + model_buffer].get_weights()
            model_buffer += 1   # activation
            if layer[5] > 0:    # pooling
                model_buffer += 1
        elif layer[0] == 3:
            continue
        elif layer[0] == 4:
            inception_weights_and_biases = []
            for j in range(0, 6):
                if j == 2:
                    continue
                inception_weights_and_biases.append(model.layers[i + model_buffer + j].get_weights())
                model_buffer += 1
            model_buffer += 1   # concatenation layer
            layer[-1] = inception_weights_and_biases
        else:
            raise NotImplementedError
        genes.overwrite_layer(layer, i)
        return genes
