
from __future__ import print_function
import numpy as np
from six.moves import cPickle as pickle
from keras.utils import plot_model
import keras as keras
from keras import backend as K
from keras.models import Sequential
from keras.losses import sparse_categorical_crossentropy
from keras.layers import Dense, Activation, Conv2D, Conv1D, MaxPooling2D, Dropout, Flatten, AveragePooling2D
from keras.optimizers import SGD

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


num_classes = 10

train_labels = keras.utils.to_categorical(train_labels, num_classes)
valid_labels = keras.utils.to_categorical(valid_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)

model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=(28, 28, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=4))
model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()


print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.1, momentum=0.01, decay=0.001),
              metrics=['accuracy'])

model.fit(train_dataset, train_labels, epochs=12, batch_size=1000, validation_data=(valid_dataset, valid_labels))
loss_and_metrics = model.evaluate(test_dataset, test_labels, batch_size=1000)
print('Test loss:', loss_and_metrics[0])
print('Test accuracy:', loss_and_metrics[1])
