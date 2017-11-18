"""
In this file we will test the encoding and train a basic network using the built structure

"""
from __future__ import print_function
from gene import *
from chromosome import *
import numpy as np
from six.moves import cPickle as pickle
import keras as keras



pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10


def reformat(dataset):
    dataset = np.expand_dims(dataset.reshape((-1, image_size, image_size)).astype(np.float32), axis=3)
    return dataset


train_dataset = reformat(train_dataset)
valid_dataset = reformat(valid_dataset)
test_dataset = reformat(test_dataset)

train_labels = keras.utils.to_categorical(train_labels, num_labels)
valid_labels = keras.utils.to_categorical(valid_labels, num_labels)
test_labels = keras.utils.to_categorical(test_labels, num_labels)

chromosome = Chromosome()
gene_config = GeneConfig("Convolutional", 32, True, (28, 28, 1), (2, 2), 'relu', 'same')
chromosome.add_gene(Gene(gene_config))
chromosome.add_gene(Gene(GeneConfig("Flatten", 0, False, 0, 0, 0, 0)))
gene_config = GeneConfig("Dense", 10, False, 0, 0, "softmax", 0)
chromosome.add_gene(Gene(gene_config))
model = chromosome.create_model()

model.compile(loss='categorical_crossentropy',
                optimizer='sgd',
                metrics=['accuracy'])

model.fit(train_dataset, train_labels, epochs=5, batch_size=100, validation_data=(valid_dataset, valid_labels))
loss_and_metrics = model.evaluate(test_dataset, test_labels, batch_size=100)
print('\nTest loss:', loss_and_metrics[0])
print('Test accuracy:', loss_and_metrics[1])
