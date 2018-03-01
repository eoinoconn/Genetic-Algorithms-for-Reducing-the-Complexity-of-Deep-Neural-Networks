from GeneticAlgorithm.utils import load_known_architecture
from keras.datasets import cifar10
from keras.utils import to_categorical, print_summary
from keras.callbacks import EarlyStopping, TensorBoard
import keras.backend as k
import logging
import configparser
import csv

img_rows = 32
img_cols = 32


def unpack_testing_data(num_labels):
    (train_dataset, train_labels), (test_dataset, test_labels) = cifar10.load_data()

    print('train_dataset shape:', train_dataset.shape)
    print(train_dataset.shape[0], 'train samples')
    print(test_dataset.shape[0], 'test samples')

    if k.image_data_format == 'channels_first':
        train_dataset = train_dataset.reshape(train_dataset.shape[0], 3, img_rows, img_cols)
        test_dataset = test_dataset.reshape(test_dataset.shape[0], 3, img_rows, img_cols)
    else:
        train_dataset = train_dataset.reshape(train_dataset.shape[0], img_rows, img_cols, 3)
        test_dataset = test_dataset.reshape(test_dataset.shape[0], img_rows, img_cols, 3)

    train_labels = to_categorical(train_labels, num_labels)
    test_labels = to_categorical(test_labels, num_labels)

    train_dataset = train_dataset.astype('float32')
    test_dataset = test_dataset.astype('float32')
    train_dataset /= 255
    test_dataset /= 255

    return {"train_dataset": train_dataset, "train_labels": train_labels,
            "valid_dataset": None, "valid_labels": None,
            "test_dataset": test_dataset, "test_labels": test_labels}


input_shape = (32, 32, 3)

chromosome = load_known_architecture('PBGen_16.csv', input_shape)
hyper_params = chromosome.hyperparameters

data = unpack_testing_data(10)

train_dataset = data['train_dataset']
train_labels = data['train_labels']
test_dataset = data['test_dataset']
test_labels = data['test_labels']

config = configparser.ConfigParser()
config.read('Config/training_parameters.ini')
# config.read('GeneticAlgorithm/Config/training_parameters.ini')

# initilise logging objects
logger_fitness = logging.getLogger('fitness')

# log geneset id
logger_fitness.info("Geneset id: %d\tAge: %d", chromosome.id, chromosome.age)

# build model
logger_fitness.info("building model")
model = chromosome.build_model()

# log geneset model
print_summary(model, print_fn=logger_fitness.info)

logger_fitness.info("Model built successfully, compiling...")

# if config['training.parameters'].getboolean('overwrite_hyperparameters'):
logger_fitness.info("Overwriting hyperparameters")
efficiency_balance = float(config['training.parameters']['efficiency_balance'])

model.compile(loss=config['training.parameters']['loss_function'],
              optimizer=config['training.parameters']['optimizer'],
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

hist = model.fit(train_dataset, train_labels,
                 epochs=100,
                 batch_size=int(config['training.parameters']['batch_size']),
                 validation_split=float(config['training.parameters']['validation_split']),
                 callbacks=callbacks,
                 verbose=int(config['training.parameters']['fit_verbose']))

# store num of model parameters
parameters = model.count_params()


if config['training.parameters'].getboolean('overwrite_hyperparameters'):
    loss_and_metrics = model.evaluate(test_dataset, test_labels,
                                      batch_size=int(config['training.parameters']['batch_size']),
                                      verbose=int(config['training.parameters']['evaluate_verbose']))
else:
    loss_and_metrics = model.evaluate(test_dataset, test_labels,
                                      batch_size=hyper_params[3],
                                      verbose=0)

accuracy = loss_and_metrics[1]

logger_fitness.info("Model evaluated successfully, fitness = %.6f, accuracy = %.6f, parameters = %d\n\n",
                        accuracy,
                        parameters)
print("Model evaluated successfully, fitness = %.6f, accuracy = %.6f, parameters = %d\n\n",
                        accuracy,
                        parameters)
