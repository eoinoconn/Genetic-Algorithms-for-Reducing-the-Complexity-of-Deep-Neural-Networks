import logging

from keras.callbacks import EarlyStopping
from keras.utils import print_summary


def assess_chromosome_fitness(genes, efficiency_balance=0.0000001,
                              train_dataset=None, train_labels=None,
                              valid_dataset=None, valid_labels=None,
                              test_dataset=None, test_labels=None):

    genes.log_geneset()

    # initilise logging objects
    logger_fitness = logging.getLogger('fitness')
    logger_genes = logging.getLogger('geneset')

    # log geneset id
    logger_fitness.info("Geneset id: %d", genes.id)

    # build model
    logger_fitness.info("building model")
    model = genes.build_model()

    # log geneset model
    print_summary(model, print_fn=logger_fitness.info)

    logger_fitness.info("Model built successfully, compiling...")

    # get hyperparameters
    hyper_params = genes.hyperparameters

    model.compile(loss=hyper_params[0],
                  optimizer=hyper_params[1],
                  metrics=['accuracy'])

    logger_fitness.info("Model compiled succesfully, beginning training...")

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.005, patience=2, verbose=0, mode='auto')

    # TB_callback = TensorBoard(log_dir='./Graph', histogram_freq=0,
    #                            write_graph=True, write_images=True)

    if valid_dataset is None:
        model.fit(train_dataset, train_labels,
                  epochs=hyper_params[2],
                  batch_size=hyper_params[3],
                  validation_split=0.16,
                  callbacks=[early_stopping],
                  verbose=0)

    else:
        model.fit(train_dataset, train_labels,
                  epochs=hyper_params[2],
                  batch_size=hyper_params[3],
                  validation_data=(valid_dataset, valid_labels),
                  callbacks=[early_stopping],
                  verbose=0)

    logger_fitness.info("Model trained succesfully, beginning evaluation...")
    loss_and_metrics = model.evaluate(test_dataset, test_labels,
                                      batch_size=hyper_params[3],
                                      verbose=0)

    # store num of model parameters
    parameters = model.count_params()
    accuracy = loss_and_metrics[1]

    fitness = cost_function(accuracy, efficiency_balance, parameters)

    logger_fitness.info("Model evaluated succesfully, fitness = %.6f, accuracy = %.6f, parameters = %d",
                        fitness,
                        accuracy,
                        parameters)
    return fitness, accuracy, parameters


def cost_function(accuracy, efficiency_balance, parameters):
    return accuracy - (efficiency_balance * parameters)

