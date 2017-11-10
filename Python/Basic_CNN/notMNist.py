# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
from six.moves import cPickle as pickle
from keras.utils import plot_model
import keras as keras
from keras import backend as K
from keras.models import Sequential
from keras.losses import sparse_categorical_crossentropy
from keras.layers import Dense, Activation, Conv2D, Conv1D, MaxPooling1D, Dropout, Flatten
from keras.optimizers import Adadelta

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    image_size = 28
    num_labels = 10


def reformat(dataset, labels):
    dataset = np.expand_dims(dataset.reshape((-1, image_size * image_size)).astype(np.float32), axis=2)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = np.expand_dims((np.arange(num_labels) == labels[:, None]).astype(np.float32), axis=2)
    return dataset, labels

if K.image_data_format() == 'channels_first':
    train_dataset = train_dataset.reshape(train_dataset.shape[0], 1, image_size, image_size)
    test_dataset = test_dataset.reshape(test_dataset.shape[0], 1, image_size, image_size)
    input_shape = (1, image_size, image_size)
else:
    train_dataset = train_dataset.reshape(train_dataset.shape[0], image_size, image_size, 1)
    test_dataset = test_dataset.reshape(test_dataset.shape[0], image_size, image_size, 1)
    input_shape = (image_size, image_size, 1)

# train_dataset, train_labels = reformat(train_dataset, train_labels)
# valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
# test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

num_classes = 10

y_train = keras.utils.to_categorical(train_labels, num_classes)
y_test = keras.utils.to_categorical(test_labels, num_classes)

model = Sequential()

#model.add(Conv1D(64, 2, activation='relu', input_shape = (784, 1)))
#model.add(MaxPooling1D(pool_size=2))
#model.add(Conv1D(32, 2, activation='relu'))
#model.add(Dense(units=10, activation='relu'))
#model.add(Dropout(0.1))
#model.add(Flatten())
#model.add(Dense(units=10, activation='softmax'))

model.add(Dense(10, input_shape=(28, 28, 1)))
# model.add(Dense(10))


if False:
    model.add(Conv1D(32, kernel_size=3,
                     activation='relu',
                     input_shape=(784,1)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adadelta(),
                  metrics=['accuracy'])

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(train_dataset, train_labels, epochs=2, batch_size=128)
loss_and_metrics = model.evaluate(train_dataset, train_labels, batch_size=512)
